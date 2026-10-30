"""Execution-layer helpers for publishing routing metadata."""

from __future__ import annotations

from typing import Any

from vela.model.manifest import PartitionManifest
from vela.model.record import VectorRecord
from vela.model.routing import RoutingEntry


def routing_entry_for_manifest(
    manifest: PartitionManifest,
    live_records: list[VectorRecord],
    manifest_key: str | None = None,
) -> RoutingEntry:
    """Build the advisory routing entry for a committed manifest."""

    return RoutingEntry(
        partition_id=manifest.partition_id,
        manifest_version=manifest.version,
        active_generation=manifest.active_generation,
        count=len(live_records),
        manifest_key=manifest_key,
        summary_covered_delta_epoch=manifest.latest_delta_epoch,
        state=manifest.state,
        centroid=_centroid(live_records),
        metadata_summary=_metadata_summary(live_records),
    )


def routing_entry_from_summary(
    manifest: PartitionManifest,
    summary: dict,
    manifest_key: str | None = None,
) -> RoutingEntry:
    """Build a routing entry from a partition-local summary object."""

    return RoutingEntry(
        partition_id=manifest.partition_id,
        manifest_version=manifest.version,
        active_generation=manifest.active_generation,
        count=summary.get("count", 0),
        manifest_key=manifest_key,
        summary_covered_delta_epoch=manifest.latest_delta_epoch,
        state=manifest.state,
        centroid=summary.get("centroid"),
        metadata_summary=summary.get("metadata_summary", {}),
    )


def _centroid(records: list[VectorRecord]) -> list[float] | None:
    if not records:
        return None
    dimension = len(records[0].vector)
    centroid = [0.0] * dimension
    for record in records:
        for idx, value in enumerate(record.vector):
            centroid[idx] += value
    return [value / len(records) for value in centroid]


def _metadata_summary(records: list[VectorRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        for key, value in record.metadata.items():
            if _is_summary_value(value):
                bucket = summary.setdefault(key, {})
                label = str(value)
                bucket[label] = bucket.get(label, 0) + 1
    return summary


def _is_summary_value(value: Any) -> bool:
    return isinstance(value, str | int | float | bool) or value is None
