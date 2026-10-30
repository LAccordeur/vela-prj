"""S3-compatible implementation of :mod:`vela.storage.object_store`."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from vela.storage.object_store import (
    ConditionalWriteFailed,
    ObjectMetadata,
    ObjectVersion,
    PointerValue,
)


class S3ObjectStore:
    """Store Vela objects in Amazon S3 or an S3-compatible service.

    MinIO deployments should provide ``endpoint_url`` and normally use
    ``addressing_style="path"``. Amazon S3 deployments can omit both and rely
    on boto3's standard region and credential resolution.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        addressing_style: str = "auto",
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket must not be empty")
        if addressing_style not in {"auto", "path", "virtual"}:
            raise ValueError("unsupported S3 addressing style")
        self.bucket = bucket
        self.prefix = _normalize_prefix(prefix)
        if client is None:
            client_options = {
                "endpoint_url": endpoint_url,
                "region_name": region_name,
                "config": Config(
                    signature_version="s3v4",
                    s3={"addressing_style": addressing_style},
                ),
            }
            credentials = {
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "aws_session_token": aws_session_token,
            }
            client_options.update({
                key: value
                for key, value in credentials.items()
                if value is not None
            })
            client = boto3.client("s3", **client_options)
        self.client = client

    def put(self, key: str, data: bytes, *, if_absent: bool = False) -> ObjectVersion:
        """Write one object, optionally requiring that its key is absent."""

        request: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._full_key(key),
            "Body": data,
        }
        if if_absent:
            request["IfNoneMatch"] = "*"
        try:
            response = self.client.put_object(**request)
        except ClientError as error:
            if if_absent and _is_conditional_failure(error):
                raise ConditionalWriteFailed(
                    f"object already exists or changed: {key}"
                ) from error
            raise
        return ObjectVersion(key=key, token=_response_etag(response))

    def get(self, key: str) -> bytes:
        """Read one complete object."""

        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
        except ClientError as error:
            if _is_missing(error):
                raise FileNotFoundError(key) from error
            raise
        return _read_body(response["Body"])

    def exists(self, key: str) -> bool:
        """Return whether one object exists."""

        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
        except ClientError as error:
            if _is_missing(error):
                return False
            raise
        return True

    def stat(self, key: str) -> ObjectMetadata:
        """Return the object's last-modified timestamp."""

        try:
            response = self.client.head_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
        except ClientError as error:
            if _is_missing(error):
                raise FileNotFoundError(key) from error
            raise
        modified = response["LastModified"]
        if not isinstance(modified, datetime):
            raise ValueError("S3 LastModified must be a datetime")
        return ObjectMetadata(key=key, last_modified=modified.timestamp())

    def delete(self, key: str) -> None:
        """Delete an object; S3 deletion is idempotent for missing keys."""

        self.client.delete_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
        )

    def list(self, prefix: str) -> list[str]:
        """List logical keys below a prefix across every S3 result page."""

        cleaned = _normalize_key(prefix, allow_empty=True)
        query_prefix = self._full_key(cleaned, allow_empty=True)
        keys: list[str] = []
        continuation: str | None = None
        while True:
            request: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": query_prefix,
            }
            if continuation is not None:
                request["ContinuationToken"] = continuation
            response = self.client.list_objects_v2(**request)
            for item in response.get("Contents", []):
                logical_key = self._logical_key(item["Key"])
                if (
                    not cleaned
                    or logical_key == cleaned
                    or logical_key.startswith(f"{cleaned}/")
                ):
                    keys.append(logical_key)
            if not response.get("IsTruncated", False):
                break
            continuation = response.get("NextContinuationToken")
            if continuation is None:
                raise ValueError("truncated S3 listing omitted continuation token")
        return sorted(set(keys))

    def read_pointer(self, key: str) -> PointerValue:
        """Read a UTF-8 pointer value and its opaque ETag compare token."""

        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
        except ClientError as error:
            if _is_missing(error):
                return PointerValue(key=key, value=None, token=None)
            raise
        value = _read_body(response["Body"]).decode("utf-8")
        return PointerValue(
            key=key,
            value=value,
            token=_response_etag(response),
        )

    def compare_and_swap_pointer(
        self,
        key: str,
        expected_token: str | None,
        new_value: str,
    ) -> PointerValue:
        """Publish a pointer using S3 ETag conditional-write semantics."""

        request: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._full_key(key),
            "Body": new_value.encode("utf-8"),
        }
        if expected_token is None:
            request["IfNoneMatch"] = "*"
        else:
            request["IfMatch"] = expected_token
        try:
            response = self.client.put_object(**request)
        except ClientError as error:
            if _is_conditional_failure(error) or (
                expected_token is not None and _is_missing(error)
            ):
                raise ConditionalWriteFailed(f"pointer changed: {key}") from error
            raise
        return PointerValue(
            key=key,
            value=new_value,
            token=_response_etag(response),
        )

    def _full_key(self, key: str, *, allow_empty: bool = False) -> str:
        cleaned = _normalize_key(key, allow_empty=allow_empty)
        if not self.prefix:
            return cleaned
        if not cleaned:
            return f"{self.prefix}/"
        return f"{self.prefix}/{cleaned}"

    def _logical_key(self, key: str) -> str:
        if not self.prefix:
            return key
        root = f"{self.prefix}/"
        if not key.startswith(root):
            raise ValueError("S3 listing returned a key outside the configured prefix")
        return key[len(root):]


def _normalize_prefix(prefix: str) -> str:
    return _normalize_key(prefix, allow_empty=True)


def _normalize_key(key: str, *, allow_empty: bool) -> str:
    cleaned = key.strip("/")
    if not cleaned:
        if allow_empty:
            return ""
        raise ValueError("object key must not be empty")
    parts = PurePosixPath(cleaned).parts
    if ".." in parts:
        raise ValueError(f"invalid object key: {key}")
    return "/".join(parts)


def _read_body(body: Any) -> bytes:
    try:
        return body.read()
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()


def _response_etag(response: dict[str, Any]) -> str:
    etag = response.get("ETag")
    if not isinstance(etag, str) or not etag:
        raise ValueError("S3 response omitted ETag")
    return etag


def _is_missing(error: ClientError) -> bool:
    return _error_status(error) == 404 or _error_code(error) in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


def _is_conditional_failure(error: ClientError) -> bool:
    return _error_status(error) in {409, 412} or _error_code(error) in {
        "409",
        "412",
        "ConditionalRequestConflict",
        "PreconditionFailed",
    }


def _error_status(error: ClientError) -> int | None:
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status if isinstance(status, int) else None


def _error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))
