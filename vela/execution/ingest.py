"""Execution-layer data ingestion."""

from __future__ import annotations

from dataclasses import dataclass

from vela.execution.generation import GenerationTransformer, transform_records
from vela.model.manifest import DeltaRef, PartitionManifest
from vela.model.record import VectorRecord
from vela.partition.policy import (
    PartitionPolicy,
    PartitionTarget,
    assign_record_batch,
)
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.utils.ids import new_commit_id, version_label


@dataclass(frozen=True)
class IngestResult:
    """Summary of an ingest attempt."""

    committed_partitions: list[str]
    conflicts: list[str]


class IngestExecutor:
    """Groups records by partition and commits one delta per touched partition."""

    def __init__(
        self,
        partition_store: PartitionStore,
        routing_store: RoutingStore,
        policy: PartitionPolicy,
        generation_transformer: GenerationTransformer | None = None,
        max_retries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.partition_store = partition_store
        self.routing_store = routing_store
        self.policy = policy
        self.generation_transformer = generation_transformer
        self.max_retries = max_retries

    def ingest(
        self,
        records: list[VectorRecord],
        tombstones: list[str] | None = None,
        candidate_partitions: list[str] | None = None,
    ) -> IngestResult:
        """Commit records and tombstones as partition-local delta epochs."""

        partitions = (
            candidate_partitions
            if candidate_partitions is not None
            else self._candidate_partitions()
        )
        grouped = self._group_records_by_leaf(records, partitions)
        grouped_tombstones: dict[str, list[str]] = {}
        for tombstone in tombstones or []:
            probe = VectorRecord(id=tombstone, vector=[])
            partition_id = self._resolve_leaf_partition(probe, partitions)
            grouped.setdefault(partition_id, [])
            grouped_tombstones.setdefault(partition_id, []).append(tombstone)

        committed: list[str] = []
        conflicts: list[str] = []
        for partition_id, partition_records in grouped.items():
            partition_tombstones = grouped_tombstones.get(partition_id, [])
            batch_committed, batch_conflicts = self._commit_partition_batch(
                partition_id,
                partition_records,
                partition_tombstones,
            )
            committed.extend(batch_committed)
            conflicts.extend(batch_conflicts)
        return IngestResult(
            committed_partitions=_unique(committed),
            conflicts=_unique(conflicts),
        )

    def _candidate_partitions(self) -> list[str]:
        view = self.routing_store.effective_view()
        if view:
            return sorted(view)
        return self.partition_store.list_partitions()

    def _commit_partition_batch(
        self,
        partition_id: str,
        records: list[VectorRecord],
        tombstones: list[str],
        source_generation: str | None = None,
    ) -> tuple[list[str], list[str]]:
        conflicts = 0
        pending_source: PartitionManifest | None = None
        pending_delta: DeltaRef | None = None
        pending_delta_epoch: int | None = None
        pending_commit_id: str | None = None
        observed_generation = source_generation
        while True:
            manifest, handle = self.partition_store.read_current_manifest(partition_id)
            if observed_generation is None:
                observed_generation = manifest.active_generation
            if manifest.state == "split":
                return self._commit_split_batch(
                    manifest,
                    records,
                    tombstones,
                    observed_generation,
                )

            can_rebase = (
                pending_source is not None
                and pending_delta is not None
                and pending_delta_epoch is not None
                and pending_commit_id is not None
                and _is_append_only_extension(pending_source, manifest)
            )
            if can_rebase:
                delta = pending_delta
                commit_id = pending_commit_id
                next_delta_epoch = max(
                    manifest.latest_delta_epoch,
                    pending_delta_epoch,
                )
            else:
                candidate_records = records
                if records and manifest.active_generation != observed_generation:
                    if self.generation_transformer is None:
                        return [], [partition_id]
                    candidate_records = transform_records(
                        records,
                        manifest.active_generation,
                        self.generation_transformer,
                    )

                commit_id = new_commit_id()
                next_delta_epoch = manifest.latest_delta_epoch + 1
                delta = self.partition_store.write_delta_bundle(
                    partition_id=partition_id,
                    generation=manifest.active_generation,
                    records=candidate_records,
                    tombstones=tombstones,
                    commit_id=commit_id,
                    epoch=version_label("e", next_delta_epoch, commit_id),
                )
                pending_source = manifest
                pending_delta = delta
                pending_delta_epoch = next_delta_epoch
                pending_commit_id = commit_id
            candidate = PartitionManifest(
                partition_id=partition_id,
                version=manifest.version + 1,
                active_generation=manifest.active_generation,
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=next_delta_epoch,
                state=manifest.state,
                base=manifest.base,
                visible_deltas=[*manifest.visible_deltas, delta],
                retained_generations=manifest.retained_generations,
                children=manifest.children,
                parent_partition_id=manifest.parent_partition_id,
                preparation_id=manifest.preparation_id,
            )
            if self.partition_store.publish_manifest(
                handle,
                candidate,
                commit_id=commit_id,
            ):
                return [partition_id], []

            if conflicts >= self.max_retries:
                return [], [partition_id]
            conflicts += 1

    def _commit_split_batch(
        self,
        manifest: PartitionManifest,
        records: list[VectorRecord],
        tombstones: list[str],
        source_generation: str,
    ) -> tuple[list[str], list[str]]:
        grouped_records = self._group_records_by_leaf(
            records,
            manifest.children,
            visited={manifest.partition_id},
        )
        grouped_tombstones: dict[str, list[str]] = {}
        for tombstone in tombstones:
            probe = VectorRecord(id=tombstone, vector=[])
            child_id = self._resolve_leaf_partition(probe, manifest.children)
            grouped_records.setdefault(child_id, [])
            grouped_tombstones.setdefault(child_id, []).append(tombstone)

        committed: list[str] = []
        conflicts: list[str] = []
        for child_id, child_records in grouped_records.items():
            child_committed, child_conflicts = self._commit_partition_batch(
                child_id,
                child_records,
                grouped_tombstones.get(child_id, []),
                source_generation,
            )
            committed.extend(child_committed)
            conflicts.extend(child_conflicts)
        return committed, conflicts

    def _group_records_by_leaf(
        self,
        records: list[VectorRecord],
        candidates: list[str],
        visited: set[str] | None = None,
    ) -> dict[str, list[VectorRecord]]:
        """Batch-place records and recursively forward assignments through splits."""

        if not records:
            return {}
        visited = set() if visited is None else set(visited)
        routing_view = self.routing_store.effective_view()
        targets = [
            PartitionTarget(
                partition_id=partition_id,
                centroid=(
                    routing_view[partition_id].centroid
                    if partition_id in routing_view
                    else None
                ),
            )
            for partition_id in candidates
        ]
        assignments = assign_record_batch(self.policy, records, targets)
        assigned: dict[str, list[VectorRecord]] = {}
        for record, partition_id in zip(records, assignments, strict=True):
            assigned.setdefault(partition_id, []).append(record)

        leaves: dict[str, list[VectorRecord]] = {}
        for partition_id, partition_records in assigned.items():
            if partition_id in visited:
                raise ValueError("split forwarding contains a partition cycle")
            manifest, _ = self.partition_store.read_current_manifest(partition_id)
            if manifest.state == "split":
                descendants = self._group_records_by_leaf(
                    partition_records,
                    manifest.children,
                    visited={*visited, partition_id},
                )
                for child_id, child_records in descendants.items():
                    leaves.setdefault(child_id, []).extend(child_records)
                continue
            self._validate_active_leaf(manifest)
            leaves.setdefault(partition_id, []).extend(partition_records)
        return leaves

    def _resolve_leaf_partition(
        self,
        record: VectorRecord,
        candidates: list[str],
    ) -> str:
        partition_id = self.policy.assign(record, candidates)
        visited: set[str] = set()
        while True:
            if partition_id in visited:
                raise ValueError("split forwarding contains a partition cycle")
            visited.add(partition_id)
            manifest, _ = self.partition_store.read_current_manifest(partition_id)
            if manifest.state == "split":
                partition_id = self.policy.assign(record, manifest.children)
                continue
            self._validate_active_leaf(manifest)
            return partition_id

    def _validate_active_leaf(self, manifest: PartitionManifest) -> None:
        if manifest.parent_partition_id is None:
            return
        parent, _ = self.partition_store.read_current_manifest(
            manifest.parent_partition_id
        )
        if parent.state != "split" or manifest.partition_id not in parent.children:
            raise ValueError(
                "child partition is not reachable from a committed split: "
                f"{manifest.partition_id}"
            )


def _is_append_only_extension(
    source: PartitionManifest,
    latest: PartitionManifest,
) -> bool:
    """Return whether an uploaded delta can be appended to the latest state."""

    source_delta_count = len(source.visible_deltas)
    return (
        source.state == "active"
        and latest.state == "active"
        and latest.partition_id == source.partition_id
        and latest.active_generation == source.active_generation
        and latest.base == source.base
        and latest.latest_base_version == source.latest_base_version
        and latest.latest_delta_epoch >= source.latest_delta_epoch
        and latest.visible_deltas[:source_delta_count] == source.visible_deltas
        and latest.retained_generations == source.retained_generations
        and latest.children == source.children
        and latest.parent_partition_id == source.parent_partition_id
        and latest.preparation_id == source.preparation_id
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
