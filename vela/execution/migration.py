"""Execution-layer embedding generation migration."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from vela.execution.generation import GenerationTransformer, transform_records
from vela.execution.routing import routing_entry_from_summary
from vela.model.manifest import PartitionManifest
from vela.model.record import VectorRecord
from vela.model.routing import RoutingEntry
from vela.storage.partition_store import BundleWriteProfile, PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.utils.ids import new_commit_id, version_label


@dataclass(frozen=True)
class MigrationPhaseTimings:
    """Accumulated phase timings for one migration, including retries."""

    read_seconds: float = 0.0
    transform_seconds: float = 0.0
    index_build_seconds: float = 0.0
    serialize_seconds: float = 0.0
    write_seconds: float = 0.0
    publish_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class MigrationResult:
    """Summary of a partition generation migration attempt."""

    partition_id: str
    source_generation: str
    target_generation: str
    committed: bool
    conflict: bool
    skipped: bool
    reason: str | None
    source_version: int
    new_version: int | None
    live_count: int
    replacement_partitions: tuple[str, ...] = ()
    timings: MigrationPhaseTimings = field(default_factory=MigrationPhaseTimings)
    prepared_manifest_key: str | None = None
    prepared_routing_entry: RoutingEntry | None = None


class MigrationExecutor:
    """Migrates one active partition to a target embedding generation."""

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

    def migrate_partition(
        self,
        partition_id: str,
        target_generation: str,
        transformer: GenerationTransformer,
        *,
        prepare_only: bool = False,
    ) -> MigrationResult:
        """Build a generation; prepare_only stages a manifest without publication.

        Staging requires a quiescent source and external coordination of visibility.
        The returned immutable manifest key is not followed by normal queries.
        """

        migration_started = time.perf_counter()
        phases = {
            name: 0.0
            for name in (
                "read",
                "transform",
                "index_build",
                "serialize",
                "write",
                "publish",
            )
        }
        read_started = time.perf_counter()
        initial, _ = self.partition_store.read_current_manifest(partition_id)
        phases["read"] += time.perf_counter() - read_started
        source_generation = initial.active_generation
        source_version = initial.version
        conflicts = 0
        last_new_version: int | None = None
        last_live_count = 0

        while True:
            read_started = time.perf_counter()
            manifest, handle = self.partition_store.read_current_manifest(partition_id)
            phases["read"] += time.perf_counter() - read_started
            if manifest.state != "active":
                return MigrationResult(
                    partition_id=partition_id,
                    source_generation=source_generation,
                    target_generation=target_generation,
                    committed=False,
                    conflict=False,
                    skipped=True,
                    reason="partition is not active",
                    source_version=source_version,
                    new_version=None,
                    live_count=0,
                    replacement_partitions=tuple(manifest.children),
                    timings=_phase_timings(phases, migration_started),
                )
            if manifest.active_generation == target_generation:
                read_started = time.perf_counter()
                live_count = len(self.partition_store.materialize_live_records(manifest))
                phases["read"] += time.perf_counter() - read_started
                return MigrationResult(
                    partition_id=partition_id,
                    source_generation=source_generation,
                    target_generation=target_generation,
                    committed=False,
                    conflict=False,
                    skipped=True,
                    reason="already on target generation",
                    source_version=source_version,
                    new_version=None,
                    live_count=live_count,
                    timings=_phase_timings(phases, migration_started),
                )

            read_started = time.perf_counter()
            source_records = self.partition_store.materialize_live_records(manifest)
            phases["read"] += time.perf_counter() - read_started
            transform_started = time.perf_counter()
            target_records = transform_records(
                source_records,
                target_generation,
                transformer,
            )
            _validate_transformed_records(source_records, target_records)
            phases["transform"] += time.perf_counter() - transform_started
            commit_id = new_commit_id()
            next_version = manifest.version + 1
            next_base_version = manifest.latest_base_version + 1
            last_new_version = next_version
            last_live_count = len(target_records)
            bundle_profile = BundleWriteProfile()
            base = self.partition_store.write_base_bundle(
                partition_id=partition_id,
                generation=target_generation,
                records=target_records,
                commit_id=commit_id,
                version=version_label("b", next_base_version, commit_id),
                profile=bundle_profile,
            )
            phases["index_build"] += bundle_profile.index_build_seconds
            phases["serialize"] += bundle_profile.serialize_seconds
            phases["write"] += bundle_profile.write_seconds
            retained_generations = dict(manifest.retained_generations)
            if manifest.base is not None:
                retained_generations[manifest.active_generation] = manifest.base
            candidate = PartitionManifest(
                partition_id=partition_id,
                version=next_version,
                active_generation=target_generation,
                latest_base_version=next_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                state=manifest.state,
                base=base,
                visible_deltas=[],
                retained_generations=retained_generations,
                children=manifest.children,
                parent_partition_id=manifest.parent_partition_id,
                preparation_id=manifest.preparation_id,
            )
            publish_started = time.perf_counter()
            if prepare_only:
                manifest_key = self.partition_store.write_manifest_candidate(candidate, commit_id)
                _, current_handle = self.partition_store.read_current_manifest(partition_id)
                if current_handle.pointer_token != handle.pointer_token:
                    raise RuntimeError("source changed during staged migration")
                summary = self.partition_store.read_summary(base.summary_key)
                routing_entry = routing_entry_from_summary(candidate, summary, manifest_key)
                phases["publish"] += time.perf_counter() - publish_started
                return MigrationResult(
                    partition_id=partition_id,
                    source_generation=source_generation,
                    target_generation=target_generation,
                    committed=False,
                    conflict=False,
                    skipped=False,
                    reason="prepared without publication",
                    source_version=manifest.version,
                    new_version=next_version,
                    live_count=len(target_records),
                    timings=_phase_timings(phases, migration_started),
                    prepared_manifest_key=manifest_key,
                    prepared_routing_entry=routing_entry,
                )
            published = self.partition_store.publish_manifest(
                handle,
                candidate,
                commit_id=commit_id,
            )
            if published:
                summary = self.partition_store.read_summary(base.summary_key)
                self.routing_store.write_partition_update(
                    routing_entry_from_summary(candidate, summary),
                    commit_id,
                )
            phases["publish"] += time.perf_counter() - publish_started
            if published:
                return MigrationResult(
                    partition_id=partition_id,
                    source_generation=source_generation,
                    target_generation=target_generation,
                    committed=True,
                    conflict=False,
                    skipped=False,
                    reason=None,
                    source_version=source_version,
                    new_version=next_version,
                    live_count=len(target_records),
                    timings=_phase_timings(phases, migration_started),
                )

            if conflicts >= self.max_retries:
                return MigrationResult(
                    partition_id=partition_id,
                    source_generation=source_generation,
                    target_generation=target_generation,
                    committed=False,
                    conflict=True,
                    skipped=False,
                    reason="manifest publish conflict after retries",
                    source_version=source_version,
                    new_version=last_new_version,
                    live_count=last_live_count,
                    timings=_phase_timings(phases, migration_started),
                )
            conflicts += 1


def _phase_timings(
    phases: dict[str, float],
    migration_started: float,
) -> MigrationPhaseTimings:
    """Freeze accumulated phase timings with end-to-end wall time."""

    return MigrationPhaseTimings(
        read_seconds=phases["read"],
        transform_seconds=phases["transform"],
        index_build_seconds=phases["index_build"],
        serialize_seconds=phases["serialize"],
        write_seconds=phases["write"],
        publish_seconds=phases["publish"],
        total_seconds=time.perf_counter() - migration_started,
    )

def _validate_transformed_records(
    source_records: list[VectorRecord],
    target_records: list[VectorRecord],
) -> None:
    if len(source_records) != len(target_records):
        raise ValueError("migration transformer must preserve record count")
    for source, target in zip(source_records, target_records):
        if source.id != target.id:
            raise ValueError("migration transformer must preserve record ids")
