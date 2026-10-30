"""Local filesystem implementation of :mod:`vela.storage.object_store`."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from vela.storage.object_store import (
    ConditionalWriteFailed,
    ObjectMetadata,
    ObjectVersion,
    PointerValue,
)


class LocalObjectStore:
    """Object store backed by files under a local root directory.

    Writers using the same root coordinate through per-key process locks.
    Object publication uses atomic filesystem operations, so readers observe
    either the old complete object or the new complete object.
    """

    _INTERNAL_DIRECTORY = ".vela-object-store"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._internal_root = self.root / self._INTERNAL_DIRECTORY
        self._lock_root = self._internal_root / "locks"
        self._temporary_root = self._internal_root / "tmp"
        self._lock_root.mkdir(parents=True, exist_ok=True)
        self._temporary_root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes, *, if_absent: bool = False) -> ObjectVersion:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._write_temporary(data)
        try:
            with self._key_lock(key):
                if if_absent:
                    try:
                        os.link(tmp, path)
                    except FileExistsError as error:
                        raise ConditionalWriteFailed(
                            f"object already exists: {key}"
                        ) from error
                else:
                    os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return ObjectVersion(key=key, token=_token(data))

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def stat(self, key: str) -> ObjectMetadata:
        path = self._path(key)
        return ObjectMetadata(key=key, last_modified=path.stat().st_mtime)

    def delete(self, key: str) -> None:
        path = self._path(key)
        with self._key_lock(key):
            path.unlink(missing_ok=True)

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix.strip("/")]
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in base.rglob("*")
            if path.is_file() and not self._is_internal(path)
        )

    def read_pointer(self, key: str) -> PointerValue:
        path = self._path(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return PointerValue(key=key, value=None, token=None)
        return PointerValue(key=key, value=data.decode("utf-8"), token=_token(data))

    def compare_and_swap_pointer(
        self,
        key: str,
        expected_token: str | None,
        new_value: str,
    ) -> PointerValue:
        data = new_value.encode("utf-8")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._write_temporary(data)
        try:
            with self._key_lock(key):
                current = self.read_pointer(key)
                if current.token != expected_token:
                    raise ConditionalWriteFailed(f"pointer changed: {key}")
                if expected_token is None:
                    try:
                        os.link(tmp, path)
                    except FileExistsError as error:
                        raise ConditionalWriteFailed(
                            f"pointer changed: {key}"
                        ) from error
                else:
                    os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return PointerValue(key=key, value=new_value, token=_token(data))

    def _path(self, key: str) -> Path:
        cleaned = key.strip("/")
        parts = Path(cleaned).parts
        if ".." in parts or (parts and parts[0] == self._INTERNAL_DIRECTORY):
            raise ValueError(f"invalid object key: {key}")
        return self.root / cleaned

    def _write_temporary(self, data: bytes) -> Path:
        path = self._temporary_root / f"{uuid4().hex}.tmp"
        with path.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        return path

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        normalized_key = key.strip("/")
        lock_name = f"{_token(normalized_key.encode('utf-8'))}.lock"
        lock_path = self._lock_root / lock_name
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _is_internal(self, path: Path) -> bool:
        relative = path.relative_to(self.root)
        return bool(relative.parts) and relative.parts[0] == self._INTERNAL_DIRECTORY


def _token(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
