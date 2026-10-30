"""Deterministic conflict-recovery tests for partition operations."""

from __future__ import annotations

import tempfile
import unittest

from vela.execution.compaction import CompactionExecutor
from vela.execution.ingest import IngestExecutor
from vela.execution.migration import MigrationExecutor
from vela.execution.split import SplitExecutor
from vela.index.flat import FlatIndex
from vela.model.manifest import PartitionManifest
from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.utils.ids import new_commit_id, version_label


class SwapTransformer:
    """Swap two-dimensional vectors when moving to another generation."""

    def transform(self, record: VectorRecord, target_generation: str) -> VectorRecord:
        return VectorRecord(
            id=record.id,
            vector=[record.vector[1], record.vector[0]],
            metadata=record.metadata,
            text=record.text,
        )


class ConflictRecoveryTest(unittest.TestCase):
    """Verify that losing operations preserve concurrently committed state."""

    def _system(self, tmpdir: str):
        index = FlatIndex()
        object_store = LocalObjectStore(tmpdir)
        partition_store = PartitionStore(object_store, index, dataset_id="test")
        routing_store = RoutingStore(object_store, dataset_id="test")
        return index, partition_store, routing_store

    def test_ingest_rebases_after_concurrent_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if candidate.partition_id == "p0000" and not injected:
                    injected = True
                    self._publish_delta(
                        partition_store,
                        "p0000",
                        [VectorRecord("winner", [0.0, 1.0])],
                        [],
                        original_publish,
                    )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [VectorRecord("loser", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )

            self.assertEqual(result.committed_partitions, ["p0000"])
            self.assertEqual(result.conflicts, [])
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.latest_delta_epoch, 1)
            self.assertEqual(len(manifest.visible_deltas), 2)
            self.assertEqual(
                len({delta.epoch for delta in manifest.visible_deltas}),
                2,
            )
            self.assertTrue(
                all(
                    delta.epoch.startswith("e000001_")
                    for delta in manifest.visible_deltas
                )
            )
            self.assertEqual(
                {record.id for record in partition_store.materialize_live_records(manifest)},
                {"winner", "loser"},
            )
            delta_vector_keys = [
                key
                for key in partition_store.store.list(
                    "datasets/test/partitions/p0000/delta"
                )
                if key.endswith("/vectors.npy")
            ]
            self.assertEqual(len(delta_vector_keys), 2)

            partition_store.publish_manifest = original_publish
            next_result = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [VectorRecord("next", [0.5, 0.5])],
                candidate_partitions=["p0000"],
            )
            self.assertEqual(next_result.conflicts, [])
            latest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(latest.latest_delta_epoch, 2)
            self.assertEqual(len(latest.visible_deltas), 3)
            self.assertTrue(latest.visible_deltas[-1].epoch.startswith("e000002_"))
            latest_delta_vector_keys = [
                key
                for key in partition_store.store.list(
                    "datasets/test/partitions/p0000/delta"
                )
                if key.endswith("/vectors.npy")
            ]
            self.assertEqual(len(latest_delta_vector_keys), 3)

    def test_ingest_fast_rebase_preserves_later_tombstone_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("doc1", [1.0, 0.0], text="base")],
            )
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if candidate.partition_id == "p0000" and not injected:
                    injected = True
                    self._publish_delta(
                        partition_store,
                        "p0000",
                        [VectorRecord("doc1", [0.0, 1.0], text="winner")],
                        [],
                        original_publish,
                    )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [],
                tombstones=["doc1"],
                candidate_partitions=["p0000"],
            )

            self.assertEqual(result.committed_partitions, ["p0000"])
            self.assertEqual(result.conflicts, [])
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.latest_delta_epoch, 1)
            self.assertEqual(len(manifest.visible_deltas), 2)
            self.assertEqual(
                partition_store.materialize_live_records(manifest),
                [],
            )

    def test_ingest_reembeds_after_generation_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if candidate.partition_id == "p0000" and not injected:
                    injected = True
                    self._publish_generation_switch(
                        partition_store,
                        "p0000",
                        "g1",
                        original_publish,
                    )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
                generation_transformer=SwapTransformer(),
            ).ingest(
                [VectorRecord("doc1", [1.0, 2.0])],
                candidate_partitions=["p0000"],
            )

            self.assertEqual(result.conflicts, [])
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.active_generation, "g1")
            records = partition_store.materialize_live_records(manifest)
            self.assertEqual(records[0].vector, [2.0, 1.0])
            self.assertTrue(
                all(delta.generation == "g1" for delta in manifest.visible_deltas)
            )

    def test_ingest_reports_conflict_when_retry_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            original_publish = partition_store.publish_manifest

            def racing_publish(handle, candidate, commit_id=None):
                current, fresh_handle = partition_store.read_current_manifest("p0000")
                competing = self._copy_manifest(
                    current,
                    version=current.version + 1,
                )
                self.assertTrue(
                    original_publish(fresh_handle, competing, new_commit_id())
                )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
                max_retries=0,
            ).ingest(
                [VectorRecord("uncommitted", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )

            self.assertEqual(result.committed_partitions, [])
            self.assertEqual(result.conflicts, ["p0000"])
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(partition_store.materialize_live_records(manifest), [])

    def test_ingest_follows_split_that_wins_publish_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if candidate.partition_id == "p0000" and not injected:
                    injected = True
                    partition_store.publish_manifest = original_publish
                    try:
                        split_result = SplitExecutor(
                            partition_store,
                            routing_store,
                        ).split_partition(
                            "p0000",
                            ["p0000a", "p0000b"],
                            HashPartitionPolicy(),
                        )
                    finally:
                        partition_store.publish_manifest = racing_publish
                    self.assertTrue(split_result.committed)
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [VectorRecord("late", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )

            self.assertEqual(result.conflicts, [])
            self.assertEqual(len(result.committed_partitions), 1)
            source, _ = partition_store.read_current_manifest("p0000")
            child_ids = {
                record.id
                for child_id in source.children
                for record in partition_store.materialize_live_records(
                    partition_store.read_current_manifest(child_id)[0]
                )
            }
            self.assertEqual(child_ids, {"late"})

    def test_compaction_preserves_delta_committed_during_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [VectorRecord("before", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if candidate.partition_id == "p0000" and not injected:
                    injected = True
                    self._publish_delta(
                        partition_store,
                        "p0000",
                        [VectorRecord("during", [0.0, 1.0])],
                        [],
                        original_publish,
                    )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = CompactionExecutor(
                partition_store,
                routing_store,
            ).compact_partition("p0000")

            self.assertTrue(result.committed)
            self.assertFalse(result.conflict)
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.base.count, 1)
            self.assertEqual(len(manifest.visible_deltas), 1)
            self.assertEqual(
                {record.id for record in partition_store.materialize_live_records(manifest)},
                {"before", "during"},
            )

    def test_migration_rebuilds_from_concurrent_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("before", [1.0, 0.0])],
            )
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if candidate.partition_id == "p0000" and not injected:
                    injected = True
                    self._publish_delta(
                        partition_store,
                        "p0000",
                        [VectorRecord("during", [0.25, 0.75])],
                        ["before"],
                        original_publish,
                    )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = MigrationExecutor(
                partition_store,
                routing_store,
            ).migrate_partition("p0000", "g1", SwapTransformer())

            self.assertTrue(result.committed)
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.active_generation, "g1")
            self.assertEqual(manifest.visible_deltas, [])
            records = {
                record.id: record
                for record in partition_store.materialize_live_records(manifest)
            }
            self.assertEqual(set(records), {"during"})
            self.assertEqual(records["during"].vector, [0.75, 0.25])

    def test_split_rebuilds_children_after_concurrent_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("before", [1.0, 0.0])],
            )
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if (
                    candidate.partition_id == "p0000"
                    and candidate.state == "split"
                    and not injected
                ):
                    injected = True
                    self._publish_delta(
                        partition_store,
                        "p0000",
                        [VectorRecord("during", [0.0, 1.0])],
                        ["before"],
                        original_publish,
                    )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = SplitExecutor(
                partition_store,
                routing_store,
            ).split_partition(
                "p0000",
                ["p0000a", "p0000b"],
                HashPartitionPolicy(),
            )

            self.assertTrue(result.committed)
            source, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(source.state, "split")
            child_manifests = [
                partition_store.read_current_manifest(child_id)[0]
                for child_id in source.children
            ]
            self.assertEqual(
                {
                    record.id
                    for child in child_manifests
                    for record in partition_store.materialize_live_records(child)
                },
                {"during"},
            )
            preparation_ids = {child.preparation_id for child in child_manifests}
            self.assertEqual(len(preparation_ids), 1)
            self.assertNotIn(None, preparation_ids)

    def test_split_rebuilds_children_after_generation_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("doc1", [1.0, 2.0])],
            )
            original_publish = partition_store.publish_manifest
            injected = False

            def racing_publish(handle, candidate, commit_id=None):
                nonlocal injected
                if (
                    candidate.partition_id == "p0000"
                    and candidate.state == "split"
                    and not injected
                ):
                    injected = True
                    self._publish_generation_switch(
                        partition_store,
                        "p0000",
                        "g1",
                        original_publish,
                    )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = SplitExecutor(
                partition_store,
                routing_store,
            ).split_partition(
                "p0000",
                ["p0000a", "p0000b"],
                HashPartitionPolicy(),
            )

            self.assertTrue(result.committed)
            source, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(source.active_generation, "g1")
            children = [
                partition_store.read_current_manifest(child_id)[0]
                for child_id in source.children
            ]
            self.assertTrue(
                all(child.active_generation == "g1" for child in children)
            )
            records = [
                record
                for child in children
                for record in partition_store.materialize_live_records(child)
            ]
            self.assertEqual(records[0].vector, [2.0, 1.0])

    def _publish_delta(
        self,
        partition_store: PartitionStore,
        partition_id: str,
        records: list[VectorRecord],
        tombstones: list[str],
        publish,
    ) -> None:
        current, handle = partition_store.read_current_manifest(partition_id)
        commit_id = new_commit_id()
        next_epoch = current.latest_delta_epoch + 1
        delta = partition_store.write_delta_bundle(
            partition_id,
            current.active_generation,
            records,
            tombstones,
            commit_id=commit_id,
            epoch=version_label("e", next_epoch, commit_id),
        )
        candidate = self._copy_manifest(
            current,
            version=current.version + 1,
            latest_delta_epoch=next_epoch,
            visible_deltas=[*current.visible_deltas, delta],
        )
        self.assertTrue(publish(handle, candidate, commit_id))

    def _publish_generation_switch(
        self,
        partition_store: PartitionStore,
        partition_id: str,
        generation: str,
        publish,
    ) -> None:
        current, handle = partition_store.read_current_manifest(partition_id)
        commit_id = new_commit_id()
        next_base_version = current.latest_base_version + 1
        source_records = partition_store.materialize_live_records(current)
        target_records = [
            SwapTransformer().transform(record, generation)
            for record in source_records
        ]
        base = partition_store.write_base_bundle(
            partition_id,
            generation,
            target_records,
            commit_id=commit_id,
            version=version_label("b", next_base_version, commit_id),
        )
        retained = dict(current.retained_generations)
        if current.base is not None:
            retained[current.active_generation] = current.base
        candidate = self._copy_manifest(
            current,
            version=current.version + 1,
            active_generation=generation,
            latest_base_version=next_base_version,
            base=base,
            visible_deltas=[],
            retained_generations=retained,
        )
        self.assertTrue(publish(handle, candidate, commit_id))

    def _copy_manifest(self, manifest: PartitionManifest, **changes) -> PartitionManifest:
        values = {
            "partition_id": manifest.partition_id,
            "version": manifest.version,
            "active_generation": manifest.active_generation,
            "latest_base_version": manifest.latest_base_version,
            "latest_delta_epoch": manifest.latest_delta_epoch,
            "state": manifest.state,
            "base": manifest.base,
            "visible_deltas": manifest.visible_deltas,
            "retained_generations": manifest.retained_generations,
            "children": manifest.children,
            "parent_partition_id": manifest.parent_partition_id,
            "preparation_id": manifest.preparation_id,
        }
        values.update(changes)
        return PartitionManifest(**values)


if __name__ == "__main__":
    unittest.main()
