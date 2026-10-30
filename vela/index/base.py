"""ANN index interface."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Protocol

from vela.model.record import VectorRecord


@dataclass(frozen=True)
class LocalIndexHit:
    """One segment-local ANN result identified by its stored row position."""

    position: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None


@dataclass(frozen=True)
class IndexBuildArtifact:
    """One serialized index plus backend-reported phase timings."""

    blob: bytes
    build_seconds: float
    serialize_seconds: float


class LoadedANNIndex(Protocol):
    """A deserialized index that can serve repeated searches."""

    count: int
    dimension: int
    metric: str

    def search(self, query: list[float], top_k: int) -> list[LocalIndexHit]:
        """Search the loaded index and return segment-local positions."""


class ANNIndex(Protocol):
    """Builds and loads one serialized ANN index object."""

    file_extension: str

    def build(self, records: list[VectorRecord]) -> bytes:
        """Build a serialized index object from records."""

    def load(self, index_blob: bytes) -> LoadedANNIndex:
        """Deserialize an index object for repeated searches."""


def build_index_profiled(
    index: ANNIndex,
    records: list[VectorRecord],
) -> IndexBuildArtifact:
    """Use backend phase profiling, with a build-only compatibility fallback."""

    method = getattr(index, "build_profiled", None)
    if method is not None:
        return method(records)
    started = time.perf_counter()
    blob = index.build(records)
    return IndexBuildArtifact(blob, time.perf_counter() - started, 0.0)
