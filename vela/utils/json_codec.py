"""JSON serialization helpers for object-store payloads."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def to_json_bytes(value: Any) -> bytes:
    """Serialize a Python value to stable JSON bytes."""

    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8")


def from_json_bytes(data: bytes) -> Any:
    """Deserialize JSON bytes."""

    return json.loads(data.decode("utf-8"))
