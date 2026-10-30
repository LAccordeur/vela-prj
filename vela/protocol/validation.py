"""Manifest validation for Vela correctness invariants."""

from __future__ import annotations

from vela.model.manifest import PartitionManifest
from vela.model.record import VectorRecord


def validate_manifest(manifest: PartitionManifest) -> None:
    """Validate basic generation and state invariants before publication."""

    if manifest.parent_partition_id == manifest.partition_id:
        raise ValueError("partition cannot be its own parent")
    if manifest.state == "active":
        if manifest.children:
            raise ValueError("active manifest must not record child partitions")
        if manifest.base and manifest.base.generation != manifest.active_generation:
            raise ValueError("active base generation must match manifest generation")
        for delta in manifest.visible_deltas:
            if delta.generation != manifest.active_generation:
                raise ValueError("visible delta generation must match manifest generation")
    elif manifest.state == "split":
        if len(manifest.children) < 2:
            raise ValueError("split manifest must record at least two child partitions")
        if len(set(manifest.children)) != len(manifest.children):
            raise ValueError("split manifest child partitions must be unique")
        if manifest.partition_id in manifest.children:
            raise ValueError("split manifest cannot include itself as a child")
    else:
        raise ValueError(f"unknown manifest state: {manifest.state}")
    for generation, retained in manifest.retained_generations.items():
        if retained.generation != generation:
            raise ValueError("retained generation key must match segment generation")


def validate_record_bundle(records: list[VectorRecord]) -> None:
    """Validate records before they are written as one immutable bundle."""

    seen: set[str] = set()
    dimension: int | None = None
    for record in records:
        if record.id in seen:
            raise ValueError(f"duplicate record id in bundle: {record.id}")
        seen.add(record.id)
        if not record.vector:
            raise ValueError(f"record vector must not be empty: {record.id}")
        if dimension is None:
            dimension = len(record.vector)
        elif len(record.vector) != dimension:
            raise ValueError("record vectors in a bundle must have the same dimension")
