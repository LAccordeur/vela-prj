"""Object-store API used by the Vela prototype.

The interface is deliberately small: immutable object writes, reads, listing,
and compare-and-swap pointer publication. The pointer CAS operation is the
storage primitive that supports the manifest correctness protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ObjectVersion:
    """Version token returned after storing an object."""

    key: str
    token: str


@dataclass(frozen=True)
class PointerValue:
    """Current value and token for a pointer object."""

    key: str
    value: str | None
    token: str | None


@dataclass(frozen=True)
class ObjectMetadata:
    """Metadata used for conservative object lifecycle decisions."""

    key: str
    last_modified: float


class ConditionalWriteFailed(RuntimeError):
    """Raised when an if-absent or compare-and-swap write fails."""


class ObjectStore(Protocol):
    """Object-store API shared by local and S3-compatible backends."""

    def put(self, key: str, data: bytes, *, if_absent: bool = False) -> ObjectVersion:
        """Write bytes, atomically requiring absence when requested."""

    def get(self, key: str) -> bytes:
        """Read bytes from a key."""

    def exists(self, key: str) -> bool:
        """Return whether a key exists."""

    def stat(self, key: str) -> ObjectMetadata:
        """Return metadata for an existing object."""

    def delete(self, key: str) -> None:
        """Delete an object if it exists."""

    def list(self, prefix: str) -> list[str]:
        """List keys under a prefix."""

    def read_pointer(self, key: str) -> PointerValue:
        """Read a small pointer object and its compare token."""

    def compare_and_swap_pointer(
        self,
        key: str,
        expected_token: str | None,
        new_value: str,
    ) -> PointerValue:
        """Atomically publish a pointer only if its token still matches."""
