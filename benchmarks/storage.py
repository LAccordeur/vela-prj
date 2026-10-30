"""Storage selection and namespace lifecycle helpers for benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
from uuid import uuid4

from vela.storage.local_object_store import LocalObjectStore
from vela.storage.object_store import ObjectStore
from vela.storage.s3_object_store import S3ObjectStore


SUPPORTED_STORAGE_BACKENDS = ("local", "s3")


def resolve_storage_namespace(
    backend: str,
    requested: str | None,
    benchmark_name: str,
) -> str | None:
    """Return a validated namespace, generating one for S3 when omitted."""

    _validate_backend(backend)
    if requested is not None:
        return _normalize_namespace(requested)
    if backend == "local":
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_normalize_namespace(benchmark_name)}/{timestamp}-{uuid4().hex[:8]}"


def create_benchmark_store(
    backend: str,
    local_root: Path,
    run_namespace: str | None,
    scenario: str,
) -> ObjectStore:
    """Create an isolated local or S3-compatible object store."""

    _validate_backend(backend)
    if backend == "local":
        return LocalObjectStore(local_root)
    if run_namespace is None:
        raise ValueError("S3 storage requires a resolved run namespace")
    bucket = os.environ.get("VELA_S3_BUCKET")
    if not bucket:
        raise ValueError("VELA_S3_BUCKET is required for S3 storage")
    return S3ObjectStore(
        bucket=bucket,
        prefix=_join_prefix(
            os.environ.get("VELA_S3_PREFIX", ""),
            run_namespace,
            scenario,
        ),
        endpoint_url=os.environ.get("VELA_S3_ENDPOINT_URL") or None,
        region_name=(
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        ),
        addressing_style=os.environ.get("VELA_S3_ADDRESSING_STYLE", "auto"),
    )


def describe_benchmark_storage(
    backend: str,
    run_namespace: str | None,
) -> dict[str, Any]:
    """Return non-secret storage details suitable for benchmark results."""

    _validate_backend(backend)
    description: dict[str, Any] = {
        "backend": backend,
        "namespace": run_namespace,
    }
    if backend == "s3":
        description.update({
            "bucket": os.environ.get("VELA_S3_BUCKET"),
            "base_prefix": os.environ.get("VELA_S3_PREFIX", ""),
            "endpoint_url": os.environ.get("VELA_S3_ENDPOINT_URL") or None,
            "addressing_style": os.environ.get(
                "VELA_S3_ADDRESSING_STYLE",
                "auto",
            ),
        })
    return description


def clone_benchmark_store(
    source: ObjectStore,
    destination: ObjectStore,
    source_root: Path,
    destination_root: Path,
) -> None:
    """Clone one isolated store outside the measured benchmark interval."""

    if isinstance(source, LocalObjectStore) and isinstance(
        destination,
        LocalObjectStore,
    ):
        shutil.copytree(source_root, destination_root, dirs_exist_ok=True)
        return
    if isinstance(source, S3ObjectStore) and isinstance(
        destination,
        S3ObjectStore,
    ):
        _copy_s3_namespace(source, destination)
        return
    raise TypeError("source and destination stores must use the same backend")


def benchmark_storage_bytes(store: ObjectStore, local_root: Path) -> int:
    """Return bytes currently stored in one benchmark namespace."""

    if isinstance(store, LocalObjectStore):
        if not local_root.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in local_root.rglob("*")
            if path.is_file()
        )
    if isinstance(store, S3ObjectStore):
        return _s3_namespace_bytes(store)
    raise TypeError(f"unsupported benchmark store: {type(store).__name__}")


def clear_benchmark_store(store: ObjectStore, local_root: Path) -> None:
    """Delete one isolated benchmark namespace."""

    if isinstance(store, LocalObjectStore):
        shutil.rmtree(local_root, ignore_errors=True)
        return
    if isinstance(store, S3ObjectStore):
        _delete_s3_namespace(store)
        return
    raise TypeError(f"unsupported benchmark store: {type(store).__name__}")


def _copy_s3_namespace(
    source: S3ObjectStore,
    destination: S3ObjectStore,
) -> None:
    source_prefix = _physical_prefix(source)
    destination_prefix = _physical_prefix(destination)
    if (
        source.bucket == destination.bucket
        and (
            source_prefix.startswith(destination_prefix)
            or destination_prefix.startswith(source_prefix)
        )
    ):
        raise ValueError("S3 clone namespaces must not overlap")

    paginator = source.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=source.bucket,
        Prefix=source_prefix,
    ):
        for item in page.get("Contents", []):
            source_key = item["Key"]
            relative_key = source_key[len(source_prefix):]
            destination.client.copy_object(
                Bucket=destination.bucket,
                Key=f"{destination_prefix}{relative_key}",
                CopySource={
                    "Bucket": source.bucket,
                    "Key": source_key,
                },
            )


def _s3_namespace_bytes(store: S3ObjectStore) -> int:
    total = 0
    paginator = store.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=store.bucket,
        Prefix=_physical_prefix(store),
    ):
        total += sum(item.get("Size", 0) for item in page.get("Contents", []))
    return total


def _delete_s3_namespace(store: S3ObjectStore) -> None:
    prefix = _physical_prefix(store)
    while True:
        page = store.client.list_objects_v2(
            Bucket=store.bucket,
            Prefix=prefix,
            MaxKeys=1000,
        )
        contents = page.get("Contents", [])
        if not contents:
            return
        response = store.client.delete_objects(
            Bucket=store.bucket,
            Delete={"Objects": [{"Key": item["Key"]} for item in contents]},
        )
        errors = response.get("Errors", [])
        if errors:
            details = ", ".join(
                f"{item.get('Key')}: {item.get('Code')}" for item in errors
            )
            raise RuntimeError(f"failed to clear S3 benchmark namespace: {details}")


def _physical_prefix(store: S3ObjectStore) -> str:
    if not store.prefix:
        raise ValueError("benchmark S3 store must have a non-empty prefix")
    return f"{store.prefix}/"


def _join_prefix(*parts: str) -> str:
    return "/".join(
        _normalize_namespace(part)
        for part in parts
        if part.strip("/")
    )


def _normalize_namespace(value: str) -> str:
    cleaned = value.strip("/")
    if not cleaned:
        raise ValueError("storage namespace must not be empty")
    parts = PurePosixPath(cleaned).parts
    if not parts or ".." in parts:
        raise ValueError(f"invalid storage namespace: {value}")
    return "/".join(parts)


def _validate_backend(backend: str) -> None:
    if backend not in SUPPORTED_STORAGE_BACKENDS:
        raise ValueError(
            f"storage backend must be one of {SUPPORTED_STORAGE_BACKENDS}"
        )
