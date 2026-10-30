"""Offline full-dataset partitioning baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math

import faiss
import numpy as np

from vela.execution.bootstrap import BootstrapExecutor
from vela.model.record import VectorRecord
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore


@dataclass(frozen=True)
class BaselineBuildResult:
    """Summary of partitions published by one offline baseline build."""

    partition_counts: dict[str, int]

    @property
    def partition_ids(self) -> list[str]:
        """Return published partition ids in deterministic order."""

        return sorted(self.partition_counts)


class BaselineExecutor:
    """Build a new Vela dataset from one complete vector-only snapshot.

    Baseline builds publish root partitions independently through
    :class:`BootstrapExecutor`; they are intended for a dataset with no existing
    partitions and do not provide dataset-wide atomic publication.
    """

    def __init__(
        self,
        partition_store: PartitionStore,
        routing_store: RoutingStore,
    ) -> None:
        self.partition_store = partition_store
        self.bootstrap = BootstrapExecutor(partition_store, routing_store)

    def build_clustering(
        self,
        records: list[VectorRecord],
        partition_count: int,
        generation: str,
        *,
        partition_prefix: str = "p",
        seed: int = 0,
        niter: int = 25,
        nredo: int = 1,
    ) -> BaselineBuildResult:
        """Cluster the complete dataset with FAISS L2 K-Means and publish it."""

        _validate_build_input(
            records,
            partition_count,
            generation,
            partition_prefix,
        )
        if niter < 1:
            raise ValueError("niter must be at least 1")
        if nredo < 1:
            raise ValueError("nredo must be at least 1")

        matrix = np.ascontiguousarray(
            [record.vector for record in records],
            dtype=np.float32,
        )

        model = faiss.Kmeans(
            matrix.shape[1],
            partition_count,
            niter=niter,
            nredo=nredo,
            seed=seed,
            verbose=False,
            gpu=False,
        )
        model.cp.min_points_per_centroid = 1
        model.train(matrix)
        _, labels = model.index.search(matrix, 1)
        grouped = [[] for _ in range(partition_count)]
        for record, label in zip(records, labels[:, 0], strict=True):
            grouped[int(label)].append(record)
        return self._publish(grouped, generation, partition_prefix)

    def build_blocks(
        self,
        records: list[VectorRecord],
        partition_count: int,
        generation: str,
        *,
        partition_prefix: str = "p",
    ) -> BaselineBuildResult:
        """Split the complete dataset into balanced contiguous blocks."""

        _validate_build_input(
            records,
            partition_count,
            generation,
            partition_prefix,
        )
        quotient, remainder = divmod(len(records), partition_count)
        grouped: list[list[VectorRecord]] = []
        offset = 0
        for index in range(partition_count):
            size = quotient + (1 if index < remainder else 0)
            grouped.append(records[offset:offset + size])
            offset += size
        return self._publish(grouped, generation, partition_prefix)

    def _publish(
        self,
        grouped: list[list[VectorRecord]],
        generation: str,
        partition_prefix: str,
    ) -> BaselineBuildResult:
        if self.partition_store.list_partitions():
            raise ValueError("baseline build requires a dataset without partitions")

        counts: dict[str, int] = {}
        width = max(4, len(str(len(grouped) - 1)))
        for index, records in enumerate(grouped):
            partition_id = f"{partition_prefix}{index:0{width}d}"
            self.bootstrap.create_partition(partition_id, generation, records)
            counts[partition_id] = len(records)
        return BaselineBuildResult(partition_counts=counts)


def _validate_build_input(
    records: list[VectorRecord],
    partition_count: int,
    generation: str,
    partition_prefix: str,
) -> None:
    if not records:
        raise ValueError("baseline dataset must not be empty")
    if partition_count < 1:
        raise ValueError("partition_count must be at least 1")
    if partition_count > len(records):
        raise ValueError("partition_count must not exceed record count")
    if not generation:
        raise ValueError("generation must not be empty")
    if not partition_prefix or "/" in partition_prefix:
        raise ValueError("partition_prefix must be non-empty and contain no slash")

    dimension = len(records[0].vector)
    if dimension == 0:
        raise ValueError("record vectors must not be empty")
    seen: set[str] = set()
    for record in records:
        if not record.id:
            raise ValueError("record id must not be empty")
        if record.id in seen:
            raise ValueError(f"duplicate record id: {record.id}")
        seen.add(record.id)
        if record.metadata or record.text is not None:
            raise ValueError("baseline records must contain vector data only")
        if len(record.vector) != dimension:
            raise ValueError("record vectors must have the same dimension")
        if any(not math.isfinite(value) for value in record.vector):
            raise ValueError("record vectors must contain only finite values")

