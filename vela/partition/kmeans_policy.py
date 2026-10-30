"""K-Means batch partition placement."""

from __future__ import annotations

import math

import faiss
import numpy as np

from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.partition.policy import PartitionTarget


class KMeansPartitionPolicy:
    """Cluster new records and place each cluster near a partition centroid.

    This is an append-oriented placement policy. Because placement depends on
    vector content and current routing summaries, it does not provide stable
    logical-id routing for upserts or tombstones.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        random_state: int = 0,
        max_iter: int = 300,
        n_init: int = 10,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self._fallback = HashPartitionPolicy()

    def assign(self, record: VectorRecord, candidates: list[str]) -> str:
        """Use stable hash placement when only a single-record API is available."""

        return self._fallback.assign(record, candidates)

    def assign_batch(
        self,
        records: list[VectorRecord],
        candidates: list[PartitionTarget],
    ) -> list[str]:
        """Cluster records and return one nearest-partition assignment per record."""

        if not candidates:
            raise ValueError("at least one candidate partition is required")
        if not records:
            return []
        _validate_records(records)
        _validate_candidates(candidates)

        cluster_count = min(self.n_clusters, len(records))
        matrix = np.ascontiguousarray(
            [record.vector for record in records],
            dtype=np.float32,
        )
        model = faiss.Kmeans(
            matrix.shape[1],
            cluster_count,
            niter=self.max_iter,
            nredo=self.n_init,
            seed=self.random_state,
            verbose=False,
            gpu=False,
        )
        model.cp.min_points_per_centroid = 1
        model.train(matrix)
        _, labels = model.index.search(matrix, 1)
        centroids = np.asarray(
            model.centroids,
            dtype=np.float32,
        ).reshape(
            cluster_count,
            matrix.shape[1],
        )
        cluster_targets = self._place_clusters(
            labels[:, 0].tolist(),
            centroids.tolist(),
            candidates,
            len(records[0].vector),
        )
        return [
            cluster_targets[int(label)]
            for label in labels[:, 0]
        ]

    def _place_clusters(
        self,
        labels: list[int],
        cluster_centroids: list[list[float]],
        candidates: list[PartitionTarget],
        dimension: int,
    ) -> dict[int, str]:
        ordered_candidates = sorted(candidates, key=lambda target: target.partition_id)
        valid = [
            target
            for target in ordered_candidates
            if _is_compatible_centroid(target.centroid, dimension)
        ]
        missing = [
            target for target in ordered_candidates if target.centroid is None
        ]
        if not valid and not missing:
            raise ValueError(
                "no candidate partition has a compatible centroid"
            )
        cluster_sizes = {
            cluster_id: labels.count(cluster_id)
            for cluster_id in range(len(cluster_centroids))
        }
        bootstrap_order = sorted(
            cluster_sizes,
            key=lambda cluster_id: (-cluster_sizes[cluster_id], cluster_id),
        )
        placements = {
            cluster_id: target.partition_id
            for cluster_id, target in zip(bootstrap_order, missing, strict=False)
        }

        for cluster_id, centroid in enumerate(cluster_centroids):
            if cluster_id in placements:
                continue
            if valid:
                placements[cluster_id] = min(
                    valid,
                    key=lambda target: (
                        _squared_distance(centroid, target.centroid or []),
                        target.partition_id,
                    ),
                ).partition_id
            else:
                placements[cluster_id] = missing[
                    cluster_id % len(missing)
                ].partition_id
        return placements


def _validate_records(records: list[VectorRecord]) -> None:
    dimension = len(records[0].vector)
    if dimension == 0:
        raise ValueError("K-Means records must contain non-empty vectors")
    if len({record.id for record in records}) != len(records):
        raise ValueError("duplicate record id in K-Means batch")
    for record in records:
        if len(record.vector) != dimension:
            raise ValueError("K-Means records must have the same dimension")
        if any(not math.isfinite(value) for value in record.vector):
            raise ValueError("K-Means vectors must contain only finite values")


def _validate_candidates(candidates: list[PartitionTarget]) -> None:
    partition_ids = [candidate.partition_id for candidate in candidates]
    if any(not partition_id for partition_id in partition_ids):
        raise ValueError("candidate partition ids must not be empty")
    if len(set(partition_ids)) != len(partition_ids):
        raise ValueError("candidate partition ids must be unique")


def _is_compatible_centroid(centroid: list[float] | None, dimension: int) -> bool:
    return (
        centroid is not None
        and len(centroid) == dimension
        and all(math.isfinite(value) for value in centroid)
    )


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    )
