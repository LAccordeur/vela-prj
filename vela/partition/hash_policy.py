"""Hash-based partition policy."""

from __future__ import annotations

import hashlib

from vela.model.record import VectorRecord
from vela.partition.policy import PartitionTarget


class HashPartitionPolicy:
    """Assign records using a stable hash of their logical id."""

    def assign(self, record: VectorRecord, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("at least one candidate partition is required")
        digest = hashlib.sha256(record.id.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % len(candidates)
        return sorted(candidates)[idx]

    def assign_batch(
        self,
        records: list[VectorRecord],
        candidates: list[PartitionTarget],
    ) -> list[str]:
        """Assign a batch while preserving record-by-record hash placement."""

        partition_ids = [candidate.partition_id for candidate in candidates]
        return [self.assign(record, partition_ids) for record in records]
