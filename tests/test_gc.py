"""Tests for garbage collection and retained object reachability."""

from __future__ import annotations

import tempfile
import unittest

from vela.execution.compaction import CompactionExecutor
from vela.execution.gc import GarbageCollector
from vela.execution.ingest import IngestExecutor
from vela.execution.migration import MigrationExecutor
from vela.execution.query import QueryExecutor
from vela.execution.split import SplitExecutor
from vela.index.flat import FlatIndex
from vela.model.manifest import PartitionManifest
from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.utils.ids import new_commit_id


class SwapTransformer:
    """Swap two-dimensional vectors for migration tests."""

    def transform(self, record: VectorRecord, target_generation: str) -> VectorRecord:
        return VectorRecord(
            id=record.id,
            vector=[record.vector[1], record.vector[0]],
            metadata=record.metadata,
            text=record.text,
        )


class PointerChangingStore(LocalObjectStore):
    """Change a pointer immediately before the GC delete-phase recheck."""

    def arm_pointer_change(self, pointer_key: str) -> None:
        self.pointer_key = pointer_key
        self.dataset_list_calls = 0
        self.armed = True

    def list(self, prefix: str) -> list[str]:
        keys = super().list(prefix)
        if getattr(self, "armed", False) and prefix == "datasets/test":
            self.dataset_list_calls += 1
            if self.dataset_list_calls == 2:
                super().put(self.pointer_key, b"raced")
                self.armed = False
        return keys


class GarbageCollectorTest(unittest.TestCase):
    """Covers dry-run, retention, grace, races, and safe sweeping."""

    def _system(self, tmpdir: str, store_type=LocalObjectStore):
        index = FlatIndex()
        object_store = store_type(tmpdir)
        partition_store = PartitionStore(object_store, index, dataset_id="test")
        routing_store = RoutingStore(object_store, dataset_id="test")
        return index, object_store, partition_store, routing_store

    def test_manifest_retention_and_dry_run_preserve_query_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            )
            ingest.ingest(
                [VectorRecord("doc1", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )
            retained_manifest, retained_handle = partition_store.read_current_manifest(
                "p0000"
            )
            retained_keys = _manifest_data_keys(retained_manifest)
            self.assertTrue(
                CompactionExecutor(partition_store, routing_store)
                .compact_partition("p0000")
                .committed
            )
            current, current_handle = partition_store.read_current_manifest("p0000")
            object_store.put("datasets/test/custom/keep.json", b"keep")
            object_store.put(
                "datasets/test/partitions/p0000/base/g0/custom/notes.txt",
                b"keep",
            )
            collector = GarbageCollector(object_store, "test")

            conservative = collector.collect(
                retain_manifest_count=2,
                grace_period_seconds=0,
            )
            self.assertIn(retained_handle.manifest_key, conservative.reachable_keys)
            self.assertTrue(retained_keys <= set(conservative.reachable_keys))

            dry_run = collector.collect(
                retain_manifest_count=1,
                grace_period_seconds=0,
            )
            self.assertIn(retained_handle.manifest_key, dry_run.garbage_keys)
            self.assertEqual(dry_run.deleted_keys, [])
            self.assertTrue(object_store.exists(retained_handle.manifest_key))

            deleted = collector.collect(
                retain_manifest_count=1,
                grace_period_seconds=0,
                dry_run=False,
            )
            self.assertIn(retained_handle.manifest_key, deleted.deleted_keys)
            self.assertTrue(object_store.exists(current_handle.manifest_key))
            self.assertTrue(_manifest_data_keys(current) <= set(deleted.reachable_keys))
            self.assertTrue(object_store.exists("datasets/test/custom/keep.json"))
            self.assertIn("datasets/test/custom/keep.json", deleted.ignored_keys)
            custom_base_key = (
                "datasets/test/partitions/p0000/base/g0/custom/notes.txt"
            )
            self.assertTrue(object_store.exists(custom_base_key))
            self.assertIn(custom_base_key, deleted.ignored_keys)
            hits = QueryExecutor(partition_store, routing_store, index).query(
                {"g0": [1.0, 0.0]},
                top_k=10,
            )
            self.assertEqual([hit.id for hit in hits], ["doc1"])

    def test_failed_manifest_publish_outputs_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, _ = self._system(tmpdir)
            manifest = partition_store.create_partition("p0000", "g0")
            _, stale = partition_store.read_current_manifest("p0000")
            _, fresh = partition_store.read_current_manifest("p0000")
            competing = PartitionManifest(
                partition_id="p0000",
                version=2,
                active_generation="g0",
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                base=manifest.base,
            )
            self.assertTrue(
                partition_store.publish_manifest(fresh, competing, commit_id="winner")
            )
            candidate_base = partition_store.write_base_bundle(
                "p0000",
                "g0",
                [VectorRecord("orphan", [0.0, 1.0])],
                commit_id="loser",
                version="b000002_loser",
            )
            candidate = PartitionManifest(
                partition_id="p0000",
                version=2,
                active_generation="g0",
                latest_base_version=2,
                latest_delta_epoch=0,
                base=candidate_base,
            )
            self.assertFalse(
                partition_store.publish_manifest(stale, candidate, commit_id="loser")
            )
            candidate_manifest_key = (
                "datasets/test/partitions/p0000/manifests/"
                "manifest_000002_loser.json"
            )
            candidate_keys = _segment_keys(candidate_base)

            report = GarbageCollector(object_store, "test").collect(
                retain_manifest_count=1,
                grace_period_seconds=0,
                dry_run=False,
            )

            self.assertIn(candidate_manifest_key, report.deleted_keys)
            self.assertTrue(candidate_keys <= set(report.deleted_keys))
            self.assertFalse(object_store.exists(candidate_manifest_key))

    def test_failed_split_children_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            )
            ingest.ingest(
                [VectorRecord("doc1", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )
            original_publish = partition_store.publish_manifest

            def racing_publish(handle, candidate, commit_id=None):
                current, fresh_handle = partition_store.read_current_manifest("p0000")
                competing = PartitionManifest(
                    partition_id="p0000",
                    version=current.version + 1,
                    active_generation=current.active_generation,
                    latest_base_version=current.latest_base_version,
                    latest_delta_epoch=current.latest_delta_epoch,
                    base=current.base,
                    visible_deltas=current.visible_deltas,
                )
                self.assertTrue(
                    original_publish(fresh_handle, competing, new_commit_id())
                )
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            result = SplitExecutor(partition_store, routing_store).split_partition(
                "p0000",
                ["p0000a", "p0000b"],
                HashPartitionPolicy(),
            )
            self.assertTrue(result.conflict)
            self.assertTrue(partition_store.partition_exists("p0000a"))

            report = GarbageCollector(object_store, "test").collect(
                retain_manifest_count=1,
                grace_period_seconds=0,
                dry_run=False,
            )

            for child_id in ("p0000a", "p0000b"):
                prefix = f"datasets/test/partitions/{child_id}"
                self.assertEqual(object_store.list(prefix), [])
                self.assertTrue(any(key.startswith(prefix) for key in report.deleted_keys))
            hits = QueryExecutor(partition_store, routing_store, index).query(
                {"g0": [1.0, 0.0]},
                top_k=10,
            )
            self.assertEqual([hit.id for hit in hits], ["doc1"])

    def test_committed_split_reclaims_source_data_after_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, object_store, partition_store, routing_store = self._system(tmpdir)
            source = partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("doc1", [1.0, 0.0])],
            )
            source_data_keys = _manifest_data_keys(source)
            result = SplitExecutor(partition_store, routing_store).split_partition(
                "p0000",
                ["p0000a", "p0000b"],
                HashPartitionPolicy(),
            )
            self.assertTrue(result.committed)

            report = GarbageCollector(object_store, "test").collect(
                retain_manifest_count=1,
                grace_period_seconds=0,
                dry_run=False,
            )

            self.assertTrue(source_data_keys <= set(report.deleted_keys))
            hits = QueryExecutor(partition_store, routing_store, index).query(
                {"g0": [1.0, 0.0]},
                top_k=10,
                candidate_partitions=["p0000"],
            )
            self.assertEqual([hit.id for hit in hits], ["doc1"])

    def test_routing_retention_removes_covered_updates_and_old_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            )
            ingest.ingest(
                [VectorRecord("doc1", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )
            updates_epoch_0 = object_store.list(
                "datasets/test/routing/partition_updates/after_000000"
            )
            self.assertTrue(routing_store.publish_merged_checkpoint(commit_id="cp1"))
            _, checkpoint_1_handle = routing_store.read_current_checkpoint()
            ingest.ingest(
                [VectorRecord("doc2", [0.0, 1.0])],
                candidate_partitions=["p0000"],
            )
            self.assertTrue(
                CompactionExecutor(partition_store, routing_store)
                .compact_partition("p0000")
                .committed
            )
            updates_epoch_1 = object_store.list(
                "datasets/test/routing/partition_updates/after_000001"
            )
            self.assertTrue(routing_store.publish_merged_checkpoint(commit_id="cp2"))
            _, checkpoint_2_handle = routing_store.read_current_checkpoint()

            report = GarbageCollector(object_store, "test").collect(
                retain_manifest_count=1,
                retain_routing_checkpoint_count=1,
                grace_period_seconds=0,
                dry_run=False,
            )

            self.assertIn(checkpoint_1_handle.checkpoint_key, report.deleted_keys)
            self.assertTrue(set(updates_epoch_0 + updates_epoch_1) <= set(report.deleted_keys))
            self.assertTrue(object_store.exists(checkpoint_2_handle.checkpoint_key))
            self.assertEqual(set(routing_store.effective_view()), {"p0000"})

    def test_retained_generation_is_a_strong_gc_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("doc1", [1.0, 0.0])],
            )
            source, _ = partition_store.read_current_manifest("p0000")
            source_base_keys = _segment_keys(source.base)
            self.assertTrue(
                MigrationExecutor(partition_store, routing_store)
                .migrate_partition("p0000", "g1", SwapTransformer())
                .committed
            )

            report = GarbageCollector(object_store, "test").collect(
                retain_manifest_count=1,
                grace_period_seconds=0,
                dry_run=False,
            )

            self.assertTrue(source_base_keys <= set(report.reachable_keys))
            self.assertTrue(all(object_store.exists(key) for key in source_base_keys))

    def test_grace_period_defers_recent_unreachable_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, _ = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            orphan = partition_store.write_base_bundle(
                "p0000",
                "g0",
                [VectorRecord("orphan", [1.0, 0.0])],
                commit_id="recent",
                version="b999999_recent",
            )
            orphan_keys = _segment_keys(orphan)

            report = GarbageCollector(object_store, "test").collect(
                retain_manifest_count=1,
                grace_period_seconds=10**9,
                dry_run=False,
            )

            self.assertTrue(orphan_keys <= set(report.deferred_keys))
            self.assertEqual(report.deleted_keys, [])
            self.assertTrue(all(object_store.exists(key) for key in orphan_keys))

    def test_pointer_change_aborts_delete_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, _ = self._system(
                tmpdir,
                store_type=PointerChangingStore,
            )
            partition_store.create_partition("p0000", "g0")
            orphan = partition_store.write_base_bundle(
                "p0000",
                "g0",
                [VectorRecord("orphan", [1.0, 0.0])],
                commit_id="orphan",
                version="b999999_orphan",
            )
            orphan_keys = _segment_keys(orphan)
            object_store.arm_pointer_change(
                "datasets/test/partitions/p0000/manifest.current"
            )

            report = GarbageCollector(object_store, "test").collect(
                retain_manifest_count=1,
                grace_period_seconds=0,
                dry_run=False,
            )

            self.assertTrue(report.aborted)
            self.assertIn("pointer changed", report.reason)
            self.assertEqual(report.deleted_keys, [])
            self.assertTrue(all(object_store.exists(key) for key in orphan_keys))


def _segment_keys(segment) -> set[str]:
    return {
        key
        for key in (
            segment.vectors_key,
            segment.ids_key,
            segment.metadata_key,
            segment.index_key,
            segment.summary_key,
        )
        if key is not None
    }


def _delta_keys(delta) -> set[str]:
    return {
        key
        for key in (
            delta.vectors_key,
            delta.ids_key,
            delta.metadata_key,
            delta.tombstones_key,
            delta.mini_index_key,
            delta.summary_delta_key,
        )
        if key is not None
    }


def _manifest_data_keys(manifest: PartitionManifest) -> set[str]:
    keys = set()
    if manifest.base is not None:
        keys.update(_segment_keys(manifest.base))
    for delta in manifest.visible_deltas:
        keys.update(_delta_keys(delta))
    for retained in manifest.retained_generations.values():
        keys.update(_segment_keys(retained))
    return keys


if __name__ == "__main__":
    unittest.main()
