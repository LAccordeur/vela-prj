"""Serialized brute-force vector index for the minimal prototype."""

from __future__ import annotations

from dataclasses import asdict
import math
import time
from typing import Any

from vela.index.base import IndexBuildArtifact, LoadedANNIndex, LocalIndexHit
from vela.model.record import VectorRecord
from vela.utils.json_codec import from_json_bytes, to_json_bytes


class FlatIndex:
    """Brute-force L2 or cosine index stored as JSON."""

    file_extension = ".json"
    _SUPPORTED_METRICS = frozenset({"l2", "cosine"})

    def __init__(self, metric: str = "l2") -> None:
        if metric not in self._SUPPORTED_METRICS:
            raise ValueError(f"unsupported flat index metric: {metric}")
        self.metric = metric

    def build(self, records: list[VectorRecord]) -> bytes:
        return self.build_profiled(records).blob

    def build_profiled(
        self,
        records: list[VectorRecord],
    ) -> IndexBuildArtifact:
        """Build the payload and measure JSON serialization separately."""

        build_started = time.perf_counter()
        payload = {
            "type": "flat",
            "metric": self.metric,
            "records": [asdict(record) for record in records],
        }
        build_seconds = time.perf_counter() - build_started
        serialize_started = time.perf_counter()
        blob = to_json_bytes(payload)
        return IndexBuildArtifact(
            blob=blob,
            build_seconds=build_seconds,
            serialize_seconds=time.perf_counter() - serialize_started,
        )

    def load(self, index_blob: bytes) -> LoadedANNIndex:
        """Parse one persisted flat index for repeated searches."""

        payload = from_json_bytes(index_blob)
        metric = payload.get("metric", "cosine")
        if metric not in self._SUPPORTED_METRICS:
            raise ValueError(f"unsupported persisted flat index metric: {metric}")
        records = payload.get("records", [])
        dimension = len(records[0]["vector"]) if records else 0
        for item in records:
            if len(item["vector"]) != dimension:
                raise ValueError("persisted flat index has mixed vector dimensions")
        return _LoadedFlatIndex(
            metric=metric,
            records=records,
            count=len(records),
            dimension=dimension,
        )


class _LoadedFlatIndex:
    """One parsed JSON flat index."""

    def __init__(
        self,
        *,
        metric: str,
        records: list[dict[str, Any]],
        count: int,
        dimension: int,
    ) -> None:
        self.metric = metric
        self.records = records
        self.count = count
        self.dimension = dimension

    def search(self, query: list[float], top_k: int) -> list[LocalIndexHit]:
        if top_k <= 0 or self.count == 0:
            return []
        if len(query) != self.dimension:
            raise ValueError("query and indexed vectors must have the same dimension")
        hits = [
            LocalIndexHit(
                position=position,
                score=_score(self.metric, query, item["vector"]),
                metadata=item.get("metadata", {}),
                text=item.get("text"),
            )
            for position, item in enumerate(self.records)
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def _score(metric: str, left: list[float], right: list[float]) -> float:
    if metric == "l2":
        return -sum(
            (a - b) ** 2
            for a, b in zip(left, right, strict=True)
        )
    return _cosine(left, right)


def _cosine(left: list[float], right: list[float]) -> float:
    denom = math.sqrt(sum(v * v for v in left)) * math.sqrt(sum(v * v for v in right))
    if denom == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denom
