"""Shared embedding-generation transformation contract."""

from __future__ import annotations

from typing import Protocol

from vela.model.record import VectorRecord


class GenerationTransformer(Protocol):
    """Transform a logical record into a requested embedding generation."""

    def transform(self, record: VectorRecord, target_generation: str) -> VectorRecord:
        """Return the target-generation representation of one logical record."""


def transform_records(
    records: list[VectorRecord],
    target_generation: str,
    transformer: GenerationTransformer,
) -> list[VectorRecord]:
    """Transform records while preserving their logical identities."""

    transformed = [
        transformer.transform(record, target_generation)
        for record in records
    ]
    for source, target in zip(records, transformed):
        if source.id != target.id:
            raise ValueError("generation transformer must preserve record ids")
    return transformed
