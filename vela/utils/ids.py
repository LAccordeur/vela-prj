"""Identifier helpers for immutable object prefixes."""

from __future__ import annotations

from uuid import uuid4


def new_commit_id() -> str:
    """Return a fresh commit id for candidate object prefixes."""

    return uuid4().hex


def version_label(prefix: str, version: int, commit_id: str) -> str:
    """Format a stable version label used in object keys."""

    return f"{prefix}{version:06d}_{commit_id}"
