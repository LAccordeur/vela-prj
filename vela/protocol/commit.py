"""Shared commit helpers for manifest-style publication."""

from __future__ import annotations

from vela.storage.object_store import ConditionalWriteFailed, ObjectStore


def publish_pointer(
    store: ObjectStore,
    pointer_key: str,
    expected_token: str | None,
    new_target_key: str,
) -> bool:
    """Conditionally publish a pointer to an immutable object key."""

    try:
        store.compare_and_swap_pointer(pointer_key, expected_token, new_target_key)
    except ConditionalWriteFailed:
        return False
    return True
