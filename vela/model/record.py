"""Record and result models used by Vela execution components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    """A logical vector object assigned to a Vela partition."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None


@dataclass(frozen=True)
class LocalHit:
    """A result produced by one ANN index object before global merge."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None


@dataclass(frozen=True)
class SearchHit:
    """A query result after partition and generation metadata are attached."""

    id: str
    score: float
    partition_id: str
    generation: str
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None
