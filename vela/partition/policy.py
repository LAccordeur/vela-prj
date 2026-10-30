"""Partition assignment policy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from vela.model.record import VectorRecord


class PartitionPolicy(Protocol):
    """Assign an incoming record to one candidate partition."""

    def assign(self, record: VectorRecord, candidates: list[str]) -> str:
        """Return a selected partition id."""


@dataclass(frozen=True)
class PartitionTarget:
    """Placement metadata for one candidate partition."""

    partition_id: str
    centroid: list[float] | None = None


@runtime_checkable
class BatchPartitionPolicy(Protocol):
    """Assign a complete record batch using candidate routing summaries."""

    def assign_batch(
        self,
        records: list[VectorRecord],
        candidates: list[PartitionTarget],
    ) -> list[str]:
        """Return one selected partition id for each input record."""


def assign_record_batch(
    policy: PartitionPolicy,
    records: list[VectorRecord],
    candidates: list[PartitionTarget],
) -> list[str]:
    """Use batch placement when supported, otherwise assign records individually."""

    if isinstance(policy, BatchPartitionPolicy):
        assignments = policy.assign_batch(records, candidates)
    else:
        partition_ids = [candidate.partition_id for candidate in candidates]
        assignments = [policy.assign(record, partition_ids) for record in records]

    if len(assignments) != len(records):
        raise ValueError("partition policy must return one assignment per record")
    candidate_ids = {candidate.partition_id for candidate in candidates}
    if any(partition_id not in candidate_ids for partition_id in assignments):
        raise ValueError("partition policy selected a non-candidate partition")
    return assignments
