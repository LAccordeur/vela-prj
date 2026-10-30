"""Execution-layer partition compaction."""

from __future__ import annotations

from dataclasses import dataclass

from vela.model.manifest import DeltaRef, PartitionManifest
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.execution.routing import routing_entry_from_summary
from vela.utils.ids import new_commit_id, version_label


@dataclass(frozen=True)
class CompactionResult:
    """Summary of a partition compaction attempt."""

    partition_id: str
    committed: bool
    conflict: bool
    skipped: bool
    reason: str | None
    source_version: int
    new_version: int | None
    compacted_delta_count: int
    live_count: int
    replacement_partitions: tuple[str, ...] = ()


class CompactionExecutor:
    """Compacts manifest-visible deltas into a fresh base bundle."""

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

    def should_compact(
        self,
        partition_id: str,
        min_delta_count: int = 1,
        min_delta_records: int = 1,
    ) -> bool:
        """Return whether the current manifest exceeds simple delta thresholds."""

        manifest, _ = self.partition_store.read_current_manifest(partition_id)
        if manifest.state != "active":
            return False
        delta_count = len(manifest.visible_deltas)
        delta_records = sum(delta.count for delta in manifest.visible_deltas)
        return delta_count >= min_delta_count and delta_records >= min_delta_records

    def compact_partition(self, partition_id: str) -> CompactionResult:
        """Fold visible deltas into a new base and publish it with manifest CAS."""

        manifest, handle = self.partition_store.read_current_manifest(partition_id)
        source_version = manifest.version
        conflicts = 0

        while True:
            skipped = self._skip_result(partition_id, manifest, source_version)
            if skipped is not None:
                return skipped

            build_source = manifest
            compacted_records = self.partition_store.materialize_live_records(build_source)
            compacted_delta_count = len(build_source.visible_deltas)
            commit_id = new_commit_id()
            next_base_version = build_source.latest_base_version + 1
            base = self.partition_store.write_base_bundle(
                partition_id=partition_id,
                generation=build_source.active_generation,
                records=compacted_records,
                commit_id=commit_id,
                version=version_label("b", next_base_version, commit_id),
            )

            while True:
                appended = _appended_deltas(build_source, manifest)
                if appended is None:
                    break
                candidate = PartitionManifest(
                    partition_id=partition_id,
                    version=manifest.version + 1,
                    active_generation=manifest.active_generation,
                    latest_base_version=next_base_version,
                    latest_delta_epoch=manifest.latest_delta_epoch,
                    state=manifest.state,
                    base=base,
                    visible_deltas=appended,
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
                    live_records = self.partition_store.materialize_live_records(candidate)
                    summary = self.partition_store.read_summary(base.summary_key)
                    self.routing_store.write_partition_update(
                        routing_entry_from_summary(candidate, summary),
                        commit_id,
                    )
                    return CompactionResult(
                        partition_id=partition_id,
                        committed=True,
                        conflict=False,
                        skipped=False,
                        reason=None,
                        source_version=source_version,
                        new_version=candidate.version,
                        compacted_delta_count=compacted_delta_count,
                        live_count=len(live_records),
                    )

                if conflicts >= self.max_retries:
                    return CompactionResult(
                        partition_id=partition_id,
                        committed=False,
                        conflict=True,
                        skipped=False,
                        reason="manifest publish conflict after retries",
                        source_version=source_version,
                        new_version=candidate.version,
                        compacted_delta_count=compacted_delta_count,
                        live_count=len(compacted_records),
                    )
                conflicts += 1
                manifest, handle = self.partition_store.read_current_manifest(
                    partition_id
                )
                if manifest.state != "active":
                    return self._skip_result(
                        partition_id,
                        manifest,
                        source_version,
                    )

            # The base or active generation changed. Rebuild from the latest
            # committed state instead of publishing assumptions from old state.

    def _skip_result(
        self,
        partition_id: str,
        manifest: PartitionManifest,
        source_version: int,
    ) -> CompactionResult | None:
        if manifest.state != "active":
            return CompactionResult(
                partition_id=partition_id,
                committed=False,
                conflict=False,
                skipped=True,
                reason="partition is not active",
                source_version=source_version,
                new_version=None,
                compacted_delta_count=len(manifest.visible_deltas),
                live_count=0,
                replacement_partitions=tuple(manifest.children),
            )
        if manifest.visible_deltas:
            return None
        live_count = len(self.partition_store.materialize_live_records(manifest))
        return CompactionResult(
            partition_id=partition_id,
            committed=False,
            conflict=False,
            skipped=True,
            reason="no visible deltas",
            source_version=source_version,
            new_version=None,
            compacted_delta_count=0,
            live_count=live_count,
        )


def _appended_deltas(
    source: PartitionManifest,
    latest: PartitionManifest,
) -> list[DeltaRef] | None:
    if (
        latest.state != "active"
        or latest.active_generation != source.active_generation
        or latest.base != source.base
        or latest.latest_base_version != source.latest_base_version
    ):
        return None
    source_count = len(source.visible_deltas)
    if latest.visible_deltas[:source_count] != source.visible_deltas:
        return None
    return latest.visible_deltas[source_count:]
