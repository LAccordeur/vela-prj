#!/usr/bin/env bash
#
# Source this file to configure Vela's S3-compatible backend:
#   source scripts/minio_env.sh
#
# Execute it to inspect the configuration or reset the isolated experiment
# prefix:
#   scripts/minio_env.sh show
#   scripts/minio_env.sh init
#   scripts/minio_env.sh reset
#   scripts/minio_env.sh reset --yes

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${MINIO_ACCESS_KEY:-minioadmin}}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${MINIO_SECRET_KEY:-minioadmin}}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"

export VELA_S3_ENDPOINT_URL="${VELA_S3_ENDPOINT_URL:-http://128.205.71.102:9000}"
export VELA_S3_BUCKET="${VELA_S3_BUCKET:-vela}"
export VELA_S3_PREFIX="${VELA_S3_PREFIX:-experiments/${USER:-unknown}}"
export VELA_S3_ADDRESSING_STYLE="${VELA_S3_ADDRESSING_STYLE:-path}"

_vela_minio_mask() {
    local value="$1"
    if [[ -z "${value}" ]]; then
        printf '<unset>'
    elif (( ${#value} <= 4 )); then
        printf '****'
    else
        printf '%s****%s' "${value:0:2}" "${value: -2}"
    fi
}

_vela_minio_show() {
    printf 'VELA_S3_ENDPOINT_URL=%s\n' "${VELA_S3_ENDPOINT_URL}"
    printf 'VELA_S3_BUCKET=%s\n' "${VELA_S3_BUCKET}"
    printf 'VELA_S3_PREFIX=%s\n' "${VELA_S3_PREFIX}"
    printf 'VELA_S3_ADDRESSING_STYLE=%s\n' "${VELA_S3_ADDRESSING_STYLE}"
    printf 'AWS_REGION=%s\n' "${AWS_REGION}"
    printf 'AWS_ACCESS_KEY_ID=%s\n' "$(_vela_minio_mask "${AWS_ACCESS_KEY_ID}")"
    printf 'AWS_SECRET_ACCESS_KEY=%s\n' \
        "$(_vela_minio_mask "${AWS_SECRET_ACCESS_KEY}")"
}

_vela_minio_python() {
    local script_dir
    local repository_root
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repository_root="$(cd -- "${script_dir}/.." && pwd)"

    if [[ -x "${repository_root}/.venv/bin/python" ]]; then
        printf '%s' "${repository_root}/.venv/bin/python"
    else
        command -v python3
    fi
}

_vela_minio_init() {
    "$(_vela_minio_python)" - <<'PY'
import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


bucket = os.environ["VELA_S3_BUCKET"]
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["VELA_S3_ENDPOINT_URL"],
    region_name=os.environ["AWS_REGION"],
    config=Config(
        s3={"addressing_style": os.environ["VELA_S3_ADDRESSING_STYLE"]}
    ),
)

try:
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket already exists: {bucket}")
    except ClientError as error:
        response = error.response
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = response.get("Error", {}).get("Code")
        if status != 404 and code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket: {bucket}")
        except ClientError as create_error:
            create_code = create_error.response.get("Error", {}).get("Code")
            if create_code != "BucketAlreadyOwnedByYou":
                raise
            print(f"Bucket already exists: {bucket}")
except (BotoCoreError, ClientError) as error:
    print(f"MinIO initialization failed: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

_vela_minio_reset() {
    local confirmation="${1:-}"
    local prefix="${VELA_S3_PREFIX#/}"
    prefix="${prefix%/}"

    if [[ -z "${prefix}" || "${prefix}" == "." ]]; then
        printf 'Refusing to reset an empty or unsafe VELA_S3_PREFIX.\n' >&2
        return 2
    fi
    if [[ -n "${confirmation}" && "${confirmation}" != "--yes" ]]; then
        printf 'Unknown reset option: %s\n' "${confirmation}" >&2
        return 2
    fi

    VELA_S3_RESET_PREFIX="${prefix}/" \
        "$(_vela_minio_python)" - "${confirmation}" <<'PY'
import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["VELA_S3_ENDPOINT_URL"],
        region_name=os.environ["AWS_REGION"],
        config=Config(
            s3={"addressing_style": os.environ["VELA_S3_ADDRESSING_STYLE"]}
        ),
    )


def summarize(client, bucket, prefix):
    count = 0
    size = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        count += len(objects)
        size += sum(item.get("Size", 0) for item in objects)
    return count, size


def delete_all(client, bucket, prefix):
    deleted = 0
    while True:
        page = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        objects = page.get("Contents", [])
        if not objects:
            return deleted
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": item["Key"]} for item in objects]},
        )
        errors = response.get("Errors", [])
        if errors:
            details = ", ".join(
                f"{item.get('Key')}: {item.get('Code')}" for item in errors
            )
            raise RuntimeError(f"failed to delete objects: {details}")
        deleted += len(objects)


bucket = os.environ["VELA_S3_BUCKET"]
prefix = os.environ["VELA_S3_RESET_PREFIX"]
confirmed = len(sys.argv) > 1 and sys.argv[1] == "--yes"

try:
    s3 = make_client()
    object_count, total_bytes = summarize(s3, bucket, prefix)
    print(
        f"Target: {os.environ['VELA_S3_ENDPOINT_URL']}/{bucket}/{prefix}"
    )
    print(f"Objects: {object_count}, bytes: {total_bytes}")
    if not confirmed:
        print("Dry run only. Re-run with 'reset --yes' to delete these objects.")
    elif object_count == 0:
        print("Nothing to delete.")
    else:
        deleted_count = delete_all(s3, bucket, prefix)
        print(f"Deleted {deleted_count} objects.")
except (BotoCoreError, ClientError, RuntimeError) as error:
    print(f"MinIO reset failed: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

_vela_minio_usage() {
    cat <<'EOF'
Usage:
  source scripts/minio_env.sh
  scripts/minio_env.sh init
  scripts/minio_env.sh show
  scripts/minio_env.sh reset [--yes]

Commands:
  init         Create VELA_S3_BUCKET if it does not already exist.
  show         Print the active MinIO configuration with masked credentials.
  reset        Preview objects under VELA_S3_PREFIX.
  reset --yes  Delete objects under VELA_S3_PREFIX.
EOF
}

# Sourcing only exports variables; command dispatch is reserved for execution.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

case "${1:-help}" in
    show)
        _vela_minio_show
        ;;
    init)
        _vela_minio_init
        ;;
    reset)
        _vela_minio_reset "${2:-}"
        ;;
    help|-h|--help)
        _vela_minio_usage
        ;;
    *)
        printf 'Unknown command: %s\n' "$1" >&2
        _vela_minio_usage >&2
        exit 2
        ;;
esac
