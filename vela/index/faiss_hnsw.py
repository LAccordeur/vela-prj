"""FAISS HNSW ANN backend with a versioned Vela binary envelope."""

from __future__ import annotations

import hashlib
import math
import struct
import time
from typing import Any

import faiss
import numpy as np

from vela.index.base import IndexBuildArtifact, LoadedANNIndex, LocalIndexHit
from vela.model.record import VectorRecord
from vela.utils.json_codec import from_json_bytes, to_json_bytes


_MAGIC = b"VELAANN1"
_PREFIX = struct.Struct("<8sI")
_FORMAT_VERSION = 1
_SUPPORTED_METRICS = frozenset({"l2", "cosine"})


class FaissHNSWIndex:
    """Build and load immutable FAISS HNSW indexes."""

    file_extension = ".faiss"

    def __init__(
        self,
        metric: str = "l2",
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(f"unsupported FAISS HNSW metric: {metric}")
        if m < 1:
            raise ValueError("HNSW m must be positive")
        if ef_construction < 1:
            raise ValueError("HNSW ef_construction must be positive")
        if ef_search < 1:
            raise ValueError("HNSW ef_search must be positive")
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, records: list[VectorRecord]) -> bytes:
        """Build an HNSW graph whose row order matches the bundle IDs object."""

        return self.build_profiled(records).blob

    def build_profiled(
        self,
        records: list[VectorRecord],
    ) -> IndexBuildArtifact:
        """Build the HNSW graph and measure its serialization separately."""

        build_started = time.perf_counter()
        matrix, dimension = _record_matrix(records)
        index = None
        if len(records) > 0:
            metric_type = (
                faiss.METRIC_L2
                if self.metric == "l2"
                else faiss.METRIC_INNER_PRODUCT
            )
            if self.metric == "cosine":
                faiss.normalize_L2(matrix)
            index = faiss.IndexHNSWFlat(dimension, self.m, metric_type)
            index.hnsw.efConstruction = self.ef_construction
            index.add(matrix)
        build_seconds = time.perf_counter() - build_started

        serialize_started = time.perf_counter()
        payload = (
            faiss.serialize_index(index).tobytes()
            if index is not None
            else b""
        )
        header = {
            "backend": "faiss_hnsw",
            "count": len(records),
            "dimension": dimension,
            "format_version": _FORMAT_VERSION,
            "metric": self.metric,
            "parameters": {
                "default_ef_search": self.ef_search,
                "ef_construction": self.ef_construction,
                "m": self.m,
            },
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        blob = _encode_envelope(header, payload)
        return IndexBuildArtifact(
            blob=blob,
            build_seconds=build_seconds,
            serialize_seconds=time.perf_counter() - serialize_started,
        )

    def load(self, index_blob: bytes) -> LoadedANNIndex:
        """Validate and deserialize one persisted HNSW index."""

        header, payload = _decode_envelope(index_blob)
        _validate_header(header, payload)
        count = header["count"]
        dimension = header["dimension"]
        metric = header["metric"]
        if count == 0:
            if payload:
                raise ValueError("empty FAISS HNSW index must not contain a payload")
            return _LoadedFaissHNSWIndex(
                index=None,
                count=0,
                dimension=dimension,
                metric=metric,
                ef_search=self.ef_search,
            )

        serialized = np.frombuffer(payload, dtype=np.uint8).copy()
        index = faiss.deserialize_index(serialized)
        if not hasattr(index, "hnsw"):
            raise ValueError("persisted FAISS index is not HNSW")
        if index.d != dimension:
            raise ValueError("FAISS index dimension does not match envelope")
        if index.ntotal != count:
            raise ValueError("FAISS index count does not match envelope")
        expected_metric = (
            faiss.METRIC_L2
            if metric == "l2"
            else faiss.METRIC_INNER_PRODUCT
        )
        if index.metric_type != expected_metric:
            raise ValueError("FAISS index metric does not match envelope")
        return _LoadedFaissHNSWIndex(
            index=index,
            count=count,
            dimension=dimension,
            metric=metric,
            ef_search=self.ef_search,
        )


class _LoadedFaissHNSWIndex:
    """One deserialized FAISS HNSW graph."""

    def __init__(
        self,
        *,
        index: Any,
        count: int,
        dimension: int,
        metric: str,
        ef_search: int,
    ) -> None:
        self.index = index
        self.count = count
        self.dimension = dimension
        self.metric = metric
        self.ef_search = ef_search

    def search(self, query: list[float], top_k: int) -> list[LocalIndexHit]:
        if top_k <= 0 or self.count == 0:
            return []
        if len(query) != self.dimension:
            raise ValueError("query and indexed vectors must have the same dimension")
        if any(not math.isfinite(value) for value in query):
            raise ValueError("query vector must contain only finite values")

        matrix = np.ascontiguousarray([query], dtype=np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(matrix)
        self.index.hnsw.efSearch = max(self.ef_search, top_k)
        distances, labels = self.index.search(matrix, min(top_k, self.count))
        hits = []
        for position, distance in zip(
            labels[0].tolist(),
            distances[0].tolist(),
            strict=True,
        ):
            if position < 0:
                continue
            score = -float(distance) if self.metric == "l2" else float(distance)
            hits.append(LocalIndexHit(position=int(position), score=score))
        return hits


def _record_matrix(records: list[VectorRecord]) -> tuple[np.ndarray, int]:
    if not records:
        return np.empty((0, 0), dtype=np.float32), 0
    dimension = len(records[0].vector)
    if dimension == 0:
        raise ValueError("record vectors must not be empty")
    for record in records:
        if len(record.vector) != dimension:
            raise ValueError("record vectors must have the same dimension")
        if any(not math.isfinite(value) for value in record.vector):
            raise ValueError("record vectors must contain only finite values")
    return (
        np.ascontiguousarray(
            [record.vector for record in records],
            dtype=np.float32,
        ),
        dimension,
    )


def _encode_envelope(header: dict[str, Any], payload: bytes) -> bytes:
    header_bytes = to_json_bytes(header)
    return _PREFIX.pack(_MAGIC, len(header_bytes)) + header_bytes + payload


def _decode_envelope(index_blob: bytes) -> tuple[dict[str, Any], bytes]:
    if len(index_blob) < _PREFIX.size:
        raise ValueError("truncated Vela ANN envelope")
    magic, header_length = _PREFIX.unpack_from(index_blob)
    if magic != _MAGIC:
        raise ValueError("invalid Vela ANN envelope magic")
    header_end = _PREFIX.size + header_length
    if header_end > len(index_blob):
        raise ValueError("truncated Vela ANN envelope header")
    header = from_json_bytes(index_blob[_PREFIX.size:header_end])
    if not isinstance(header, dict):
        raise ValueError("Vela ANN envelope header must be an object")
    return header, index_blob[header_end:]


def _validate_header(header: dict[str, Any], payload: bytes) -> None:
    required = {
        "backend",
        "count",
        "dimension",
        "format_version",
        "metric",
        "payload_sha256",
    }
    if not required.issubset(header):
        raise ValueError("Vela ANN envelope header is missing required fields")
    if header["format_version"] != _FORMAT_VERSION:
        raise ValueError("unsupported Vela ANN envelope version")
    if header["backend"] != "faiss_hnsw":
        raise ValueError("Vela ANN envelope backend is not faiss_hnsw")
    if header["metric"] not in _SUPPORTED_METRICS:
        raise ValueError("unsupported metric in Vela ANN envelope")
    if (
        not isinstance(header["count"], int)
        or header["count"] < 0
        or not isinstance(header["dimension"], int)
        or header["dimension"] < 0
    ):
        raise ValueError("invalid count or dimension in Vela ANN envelope")
    actual_hash = hashlib.sha256(payload).hexdigest()
    if header["payload_sha256"] != actual_hash:
        raise ValueError("FAISS HNSW payload checksum mismatch")
