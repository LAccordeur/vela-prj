"""Execution-layer partition splitting."""

from __future__ import annotations

from dataclasses import dataclass

from vela.execution.routing import routing_entry_for_manifest, routing_entry_from_summary
from vela.model.manifest import PartitionManifest
from vela.partition.policy import (
    PartitionPolicy,
    PartitionTarget,
    assign_record_batch,
)
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.utils.ids import new_commit_id


@dataclass(frozen=True)
class SplitResult:
    """Summary of one partition split attempt."""

    partition_id: str
    child_partition_ids: list[str]
    committed: bool
    conflict: bool
    skipped: bool
    reason: str | None
    source_version: int
    new_version: int | None
    live_count: int
    child_counts: dict[str, int]


class SplitExecutor:
    """Replace one active source partition with active child partitions."""

    def __init__(
        self,
        partition_store: PartitionStore,
        routing_store: RoutingStore,
        max_retries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.partition_store = partition_store
        self.routing_store = routing_store
        self.max_retries = max_retries

    def split_partition(
        self,
        partition_id: str,
        child_partition_ids: list[str],
        policy: PartitionPolicy,
    ) -> SplitResult:
        """Materialize, partition, and publish children behind a split manifest."""

        children = _validate_child_partition_ids(partition_id, child_partition_ids)
        initial, _ = self.partition_store.read_current_manifest(partition_id)
        source_version = initial.version
        preparation_id = new_commit_id()
        conflicts = 0
        last_new_version: int | None = None
        last_live_count = 0
        last_child_counts: dict[str, int] = {}

        while True:
            manifest, handle = self.partition_store.read_current_manifest(partition_id)
            if manifest.state != "active":
                return SplitResult(
                    partition_id=partition_id,
                    child_partition_ids=children,
                    committed=False,
                    conflict=False,
                    skipped=True,
                    reason="partition is not active",
                    source_version=source_version,
                    new_version=None,
                    live_count=0,
                    child_counts={},
                )

            live_records = self.partition_store.materialize_live_records(manifest)
            grouped = {child_id: [] for child_id in children}
            assignments = assign_record_batch(
                policy,
                live_records,
                [PartitionTarget(child_id) for child_id in children],
            )
            for record, child_id in zip(live_records, assignments, strict=True):
                grouped[child_id].append(record)
            child_counts = {
                child_id: len(records)
                for child_id, records in grouped.items()
            }
            last_live_count = len(live_records)
            last_child_counts = child_counts
            child_manifests = {
                child_id: self.partition_store.prepare_split_child(
                    child_id,
                    manifest.active_generation,
                    grouped[child_id],
                    parent_partition_id=partition_id,
                    preparation_id=preparation_id,
                )
                for child_id in children
            }

            commit_id = new_commit_id()
            next_version = manifest.version + 1
            last_new_version = next_version
            candidate = PartitionManifest(
                partition_id=partition_id,
                version=next_version,
                active_generation=manifest.active_generation,
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                state="split",
                base=manifest.base,
                visible_deltas=manifest.visible_deltas,
                retained_generations=manifest.retained_generations,
                children=children,
                parent_partition_id=manifest.parent_partition_id,
                preparation_id=manifest.preparation_id,
            )
            if self.partition_store.publish_manifest(
                handle,
                candidate,
                commit_id=commit_id,
            ):
                source_entry = routing_entry_for_manifest(candidate, [])
                child_entries = {
                    child_id: routing_entry_from_summary(
                        child_manifest,
                        self.partition_store.read_base_summary(child_manifest),
                    )
                    for child_id, child_manifest in child_manifests.items()
                }
                self.routing_store.write_split_update(
                    source_entry,
                    child_entries,
                    commit_id=commit_id,
                )
                return SplitResult(
                    partition_id=partition_id,
                    child_partition_ids=children,
                    committed=True,
                    conflict=False,
                    skipped=False,
                    reason=None,
                    source_version=source_version,
                    new_version=next_version,
                    live_count=len(live_records),
                    child_counts=child_counts,
                )

            if conflicts >= self.max_retries:
                return SplitResult(
                    partition_id=partition_id,
                    child_partition_ids=children,
                    committed=False,
                    conflict=True,
                    skipped=False,
                    reason="manifest publish conflict after retries",
                    source_version=source_version,
                    new_version=last_new_version,
                    live_count=last_live_count,
                    child_counts=last_child_counts,
                )
            conflicts += 1


def _validate_child_partition_ids(
    source_partition_id: str,
    child_partition_ids: list[str],
) -> list[str]:
    children = list(child_partition_ids)
    if len(children) < 2:
        raise ValueError("partition split requires at least two child partitions")
    if any(not child_id for child_id in children):
        raise ValueError("child partition ids must not be empty")
    if len(set(children)) != len(children):
        raise ValueError("child partition ids must be unique")
    if source_partition_id in children:
        raise ValueError("source partition cannot be one of its children")
    return children
