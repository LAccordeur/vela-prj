"""Tests for the minimal Vela prototype."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from vela.execution.bootstrap import BootstrapExecutor
from vela.execution.compaction import CompactionExecutor
from vela.execution.ingest import IngestExecutor
from vela.execution.migration import MigrationExecutor
from vela.execution.query import QueryExecutor
from vela.execution.split import SplitExecutor
from vela.index.flat import FlatIndex
from vela.model.manifest import ManifestHandle, PartitionManifest
from vela.model.record import VectorRecord
from vela.model.routing import RoutingCheckpoint, RoutingCheckpointHandle, RoutingEntry
from vela.partition.hash_policy import HashPartitionPolicy
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore, ReferencedObjectMissing
from vela.storage.routing_store import RoutingStore
from vela.utils.ids import new_commit_id


class SwapGenerationTransformer:
    """Test transformer that swaps two-dimensional vectors for a new generation."""

    def transform(self, record: VectorRecord, target_generation: str) -> VectorRecord:
        return VectorRecord(
            id=record.id,
            vector=[record.vector[1], record.vector[0]],
            metadata=record.metadata,
            text=record.text,
        )


class EmptyVectorTransformer:
    """Test transformer that emits invalid vectors."""

    def transform(self, record: VectorRecord, target_generation: str) -> VectorRecord:
        return VectorRecord(record.id, [], record.metadata, record.text)


class StaleFirstPartitionStore:
    """Test proxy that returns one stale manifest before delegating reads."""

    def __init__(self, delegate: PartitionStore, stale_manifest: PartitionManifest) -> None:
        self.delegate = delegate
        self.stale_manifest = stale_manifest
        self.served_stale = False

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def read_current_manifest(self, partition_id: str):
        if (
            partition_id == self.stale_manifest.partition_id
            and not self.served_stale
        ):
            self.served_stale = True
            return self.stale_manifest, ManifestHandle(
                partition_id=partition_id,
                manifest_key="stale",
                version=self.stale_manifest.version,
                pointer_token="stale",
            )
        return self.delegate.read_current_manifest(partition_id)


class RecordingPolicy:
    """Test policy that records candidate partitions passed by ingest."""

    def __init__(self) -> None:
        self.candidates: list[list[str]] = []

    def assign(self, record: VectorRecord, candidates: list[str]) -> str:
        self.candidates.append(list(candidates))
        if not candidates:
            raise ValueError("at least one candidate partition is required")
        return candidates[0]


class RacingRoutingStore(RoutingStore):
    """Test store that simulates checkpoint advancement during update writes."""

    def __init__(self, object_store, dataset_id: str, latest_checkpoint: RoutingCheckpoint) -> None:
        super().__init__(object_store, dataset_id)
        self.latest_checkpoint = latest_checkpoint
        self.reads = 0

    def read_current_checkpoint(self):
        self.reads += 1
        if self.reads == 1:
            return RoutingCheckpoint(epoch=0), RoutingCheckpointHandle(None, 0, None)
        return self.latest_checkpoint, RoutingCheckpointHandle("latest", self.latest_checkpoint.epoch, "token")


class PrototypeTest(unittest.TestCase):
    """Covers storage visibility and execution-layer behavior."""

    def _system(self, tmpdir: str):
        index = FlatIndex()
        object_store = LocalObjectStore(tmpdir)
        partition_store = PartitionStore(object_store, index, dataset_id="test")
        routing_store = RoutingStore(object_store, dataset_id="test")
        return index, object_store, partition_store, routing_store

    def test_unpublished_delta_is_not_query_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            partition_store.write_delta_bundle(
                "p0000",
                "g0",
                [VectorRecord("hidden", [1.0, 0.0])],
                [],
                commit_id=new_commit_id(),
            )
            query = QueryExecutor(partition_store, routing_store, index)

            self.assertEqual(query.query({"g0": [1.0, 0.0]}, top_k=10), [])

    def test_vector_only_bundles_omit_metadata_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("base", [1.0, 0.0])],
            )
            result = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [VectorRecord("delta", [0.0, 1.0])],
                candidate_partitions=["p0000"],
            )

            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertIsNotNone(manifest.base)
            self.assertIsNone(manifest.base.metadata_key)
            self.assertEqual(result.conflicts, [])
            self.assertEqual(len(manifest.visible_deltas), 1)
            self.assertIsNone(manifest.visible_deltas[0].metadata_key)
            self.assertFalse(any(
                key.endswith("/metadata.json")
                for key in object_store.list(
                    "datasets/test/partitions/p0000"
                )
            ))
            self.assertEqual(
                partition_store.materialize_live_records(manifest),
                [
                    VectorRecord("base", [1.0, 0.0]),
                    VectorRecord("delta", [0.0, 1.0]),
                ],
            )

    def test_bundles_preserve_nonempty_metadata_and_text_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, routing_store = self._system(tmpdir)
            base_record = VectorRecord(
                "base",
                [1.0, 0.0],
                metadata={"tenant": "a"},
            )
            delta_record = VectorRecord(
                "delta",
                [0.0, 1.0],
                text="searchable text",
            )
            partition_store.create_partition("p0000", "g0", [base_record])
            IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [delta_record],
                candidate_partitions=["p0000"],
            )

            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertIsNotNone(manifest.base)
            self.assertIsNotNone(manifest.base.metadata_key)
            self.assertTrue(object_store.exists(manifest.base.metadata_key))
            self.assertEqual(len(manifest.visible_deltas), 1)
            delta = manifest.visible_deltas[0]
            self.assertIsNotNone(delta.metadata_key)
            self.assertTrue(object_store.exists(delta.metadata_key))
            self.assertEqual(
                partition_store.materialize_live_records(manifest),
                [base_record, delta_record],
            )

    def test_bootstrap_create_partition_publishes_routing_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            bootstrap = BootstrapExecutor(partition_store, routing_store)

            manifest = bootstrap.create_partition("p0000", "g0")

            self.assertEqual(manifest.version, 1)
            view = routing_store.effective_view()
            self.assertIn("p0000", view)
            self.assertEqual(view["p0000"].manifest_version, 1)
            self.assertEqual(view["p0000"].active_generation, "g0")
            self.assertEqual(view["p0000"].count, 0)

    def test_ingest_publishes_delta_and_query_reads_manifest_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            result = ingest.ingest([VectorRecord("doc1", [1.0, 0.0], text="visible")])

            self.assertEqual(result.conflicts, [])
            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query({"g0": [1.0, 0.0]}, top_k=1)
            self.assertEqual(hits[0].id, "doc1")

    def test_ingest_uses_routing_view_for_default_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            BootstrapExecutor(partition_store, routing_store).create_partition("p0001", "g0")
            policy = RecordingPolicy()
            ingest = IngestExecutor(partition_store, routing_store, policy)

            result = ingest.ingest([VectorRecord("doc1", [1.0, 0.0])])

            self.assertEqual(result.committed_partitions, ["p0001"])
            self.assertEqual(policy.candidates, [["p0001"]])

    def test_ingest_falls_back_to_list_partitions_when_routing_view_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            policy = RecordingPolicy()
            ingest = IngestExecutor(partition_store, routing_store, policy)

            result = ingest.ingest([VectorRecord("doc1", [1.0, 0.0])])

            self.assertEqual(result.committed_partitions, ["p0000"])
            self.assertEqual(policy.candidates, [["p0000"]])

    def test_ingest_explicit_candidates_override_routing_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            BootstrapExecutor(partition_store, routing_store).create_partition("p0001", "g0")
            policy = RecordingPolicy()
            ingest = IngestExecutor(partition_store, routing_store, policy)

            result = ingest.ingest(
                [VectorRecord("doc1", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )

            self.assertEqual(result.committed_partitions, ["p0000"])
            self.assertEqual(policy.candidates, [["p0000"]])

    def test_query_retries_with_latest_manifest_when_referenced_object_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])], candidate_partitions=["p0000"])
            stale_manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertTrue(stale_manifest.visible_deltas)

            self.assertTrue(
                CompactionExecutor(partition_store, routing_store)
                .compact_partition("p0000")
                .committed
            )
            object_store.delete(stale_manifest.visible_deltas[0].mini_index_key)

            query = QueryExecutor(
                StaleFirstPartitionStore(partition_store, stale_manifest),
                routing_store,
                index,
                max_missing_retries=2,
            )

            hits = query.query(
                {"g0": [1.0, 0.0]},
                top_k=1,
                candidate_partitions=["p0000"],
            )

            self.assertEqual([hit.id for hit in hits], ["doc1"])

    def test_query_raises_after_missing_object_retry_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, object_store, partition_store, routing_store = self._system(tmpdir)
            manifest = partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("doc1", [1.0, 0.0])],
            )
            self.assertIsNotNone(manifest.base)
            object_store.delete(manifest.base.index_key)
            query = QueryExecutor(
                partition_store,
                routing_store,
                index,
                max_missing_retries=1,
            )

            with self.assertRaises(ReferencedObjectMissing):
                query.query(
                    {"g0": [1.0, 0.0]},
                    top_k=1,
                    candidate_partitions=["p0000"],
                )

    def test_manifest_base_and_delta_versions_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            manifest = partition_store.create_partition("p0000", "g0")
            self.assertEqual(manifest.version, 1)
            self.assertEqual(manifest.latest_base_version, 1)
            self.assertEqual(manifest.latest_delta_epoch, 0)
            self.assertIsNotNone(manifest.base)
            self.assertTrue(manifest.base.version.startswith("b000001_"))

            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])], candidate_partitions=["p0000"])
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.version, 2)
            self.assertEqual(manifest.latest_base_version, 1)
            self.assertEqual(manifest.latest_delta_epoch, 1)
            self.assertEqual(len(manifest.visible_deltas), 1)
            self.assertTrue(manifest.visible_deltas[-1].epoch.startswith("e000001_"))

            ingest.ingest([VectorRecord("doc2", [0.0, 1.0])], candidate_partitions=["p0000"])
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.version, 3)
            self.assertEqual(manifest.latest_base_version, 1)
            self.assertEqual(manifest.latest_delta_epoch, 2)
            self.assertEqual(len(manifest.visible_deltas), 2)
            self.assertTrue(manifest.visible_deltas[-1].epoch.startswith("e000002_"))

    def test_manifest_cas_conflict_rejects_stale_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, _ = self._system(tmpdir)
            manifest = partition_store.create_partition("p0000", "g0")
            _, stale = partition_store.read_current_manifest("p0000")
            _, fresh = partition_store.read_current_manifest("p0000")

            next_manifest = PartitionManifest(
                partition_id="p0000",
                version=manifest.version + 1,
                active_generation="g0",
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                base=manifest.base,
            )
            self.assertTrue(partition_store.publish_manifest(fresh, next_manifest))

            stale_candidate = PartitionManifest(
                partition_id="p0000",
                version=manifest.version + 1,
                active_generation="g0",
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                base=manifest.base,
            )
            self.assertFalse(partition_store.publish_manifest(stale, stale_candidate))


    def test_upsert_returns_latest_record_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0], text="old")])
            ingest.ingest([
                VectorRecord("doc1", [0.0, 1.0], {"version": "new"}, "new")
            ])

            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query({"g0": [1.0, 0.0]}, top_k=10)

            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].id, "doc1")
            self.assertEqual(hits[0].text, "new")
            self.assertEqual(hits[0].metadata, {"version": "new"})

    def test_failed_stale_publish_does_not_make_candidate_delta_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            manifest = partition_store.create_partition("p0000", "g0")
            _, stale = partition_store.read_current_manifest("p0000")
            _, fresh = partition_store.read_current_manifest("p0000")
            hidden_delta = partition_store.write_delta_bundle(
                "p0000",
                "g0",
                [VectorRecord("hidden", [1.0, 0.0])],
                [],
                commit_id=new_commit_id(),
            )
            fresh_candidate = PartitionManifest(
                partition_id="p0000",
                version=manifest.version + 1,
                active_generation="g0",
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                base=manifest.base,
            )
            stale_candidate = PartitionManifest(
                partition_id="p0000",
                version=manifest.version + 1,
                active_generation="g0",
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch + 1,
                base=manifest.base,
                visible_deltas=[hidden_delta],
            )

            self.assertTrue(partition_store.publish_manifest(fresh, fresh_candidate))
            self.assertFalse(partition_store.publish_manifest(stale, stale_candidate))

            query = QueryExecutor(partition_store, routing_store, index)
            self.assertEqual(query.query({"g0": [1.0, 0.0]}, top_k=10), [])

    def test_record_bundle_validation_rejects_invalid_delta_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())

            with self.assertRaisesRegex(ValueError, "duplicate record id"):
                ingest.ingest([
                    VectorRecord("doc1", [1.0, 0.0]),
                    VectorRecord("doc1", [0.0, 1.0]),
                ])
            with self.assertRaisesRegex(ValueError, "same dimension"):
                ingest.ingest([
                    VectorRecord("doc1", [1.0, 0.0]),
                    VectorRecord("doc2", [1.0]),
                ])
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                ingest.ingest([VectorRecord("doc1", [])])

    def test_ingest_does_not_refresh_routing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            BootstrapExecutor(partition_store, routing_store).create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())

            ingest.ingest([
                VectorRecord("doc1", [1.0, 0.0]),
                VectorRecord("doc2", [0.0, 1.0]),
            ])
            ingest.ingest([VectorRecord("doc1", [0.5, 0.5])])
            ingest.ingest([], tombstones=["doc2"])

            view = routing_store.effective_view()
            self.assertEqual(view["p0000"].count, 0)


    def test_compaction_folds_visible_deltas_into_new_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([
                VectorRecord("doc1", [1.0, 0.0], text="old"),
                VectorRecord("doc2", [0.0, 1.0]),
            ])
            ingest.ingest([
                VectorRecord("doc1", [0.5, 0.5], {"version": "new"}, "new")
            ])
            ingest.ingest([], tombstones=["doc2"])

            compaction = CompactionExecutor(partition_store, routing_store)
            result = compaction.compact_partition("p0000")

            self.assertTrue(result.committed)
            self.assertFalse(result.conflict)
            self.assertEqual(result.compacted_delta_count, 3)
            self.assertEqual(result.live_count, 1)
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.version, 5)
            self.assertEqual(manifest.latest_base_version, 2)
            self.assertEqual(manifest.latest_delta_epoch, 3)
            self.assertEqual(manifest.visible_deltas, [])
            self.assertIsNotNone(manifest.base)
            self.assertEqual(manifest.base.count, 1)
            self.assertTrue(manifest.base.version.startswith("b000002_"))

            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query({"g0": [1.0, 0.0]}, top_k=10)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].id, "doc1")
            self.assertEqual(hits[0].text, "new")
            self.assertEqual(routing_store.effective_view()["p0000"].count, 1)

    def test_compaction_skips_partition_without_visible_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            compaction = CompactionExecutor(partition_store, routing_store)

            result = compaction.compact_partition("p0000")

            self.assertFalse(result.committed)
            self.assertTrue(result.skipped)
            self.assertEqual(result.reason, "no visible deltas")
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.version, 1)

    def test_compaction_skips_split_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            manifest = partition_store.create_partition("p0000", "g0")
            _, handle = partition_store.read_current_manifest("p0000")
            split_manifest = PartitionManifest(
                partition_id="p0000",
                version=manifest.version + 1,
                active_generation="g0",
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                state="split",
                base=manifest.base,
                children=["p0000a", "p0000b"],
            )
            self.assertTrue(partition_store.publish_manifest(handle, split_manifest))
            compaction = CompactionExecutor(partition_store, routing_store)

            result = compaction.compact_partition("p0000")

            self.assertFalse(result.committed)
            self.assertTrue(result.skipped)
            self.assertEqual(result.reason, "partition is not active")

    def test_compaction_reports_manifest_publish_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])])
            original_publish = partition_store.publish_manifest

            def racing_publish(handle, candidate, commit_id=None):
                current, fresh_handle = partition_store.read_current_manifest("p0000")
                competing = PartitionManifest(
                    partition_id="p0000",
                    version=current.version + 1,
                    active_generation=current.active_generation,
                    latest_base_version=current.latest_base_version,
                    latest_delta_epoch=current.latest_delta_epoch,
                    state=current.state,
                    base=current.base,
                    visible_deltas=current.visible_deltas,
                    retained_generations=current.retained_generations,
                    children=current.children,
                )
                self.assertTrue(original_publish(fresh_handle, competing, new_commit_id()))
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            compaction = CompactionExecutor(partition_store, routing_store)

            result = compaction.compact_partition("p0000")

            self.assertFalse(result.committed)
            self.assertTrue(result.conflict)
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(len(manifest.visible_deltas), 1)
            query = QueryExecutor(partition_store, routing_store, index)
            self.assertEqual(query.query({"g0": [1.0, 0.0]}, top_k=1)[0].id, "doc1")

    def test_should_compact_uses_delta_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            compaction = CompactionExecutor(partition_store, routing_store)
            self.assertFalse(compaction.should_compact("p0000"))

            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])])

            self.assertTrue(compaction.should_compact("p0000"))
            self.assertFalse(compaction.should_compact("p0000", min_delta_count=2))
            self.assertFalse(compaction.should_compact("p0000", min_delta_records=2))


    def test_partition_summary_contains_metadata_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)

            manifest = BootstrapExecutor(partition_store, routing_store).create_partition(
                "p0000",
                "g0",
                [
                    VectorRecord("doc1", [1.0, 3.0], {"tenant": "a", "project": "x"}),
                    VectorRecord("doc2", [3.0, 1.0], {"tenant": "a", "project": "y"}),
                ],
            )

            summary = partition_store.read_base_summary(manifest)
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["centroid"], [2.0, 2.0])
            self.assertEqual(summary["metadata_summary"]["tenant"], {"a": 2})
            self.assertEqual(summary["metadata_summary"]["project"], {"x": 1, "y": 1})
            entry = routing_store.effective_view()["p0000"]
            self.assertEqual(entry.count, summary["count"])
            self.assertEqual(entry.centroid, summary["centroid"])
            self.assertEqual(entry.metadata_summary, summary["metadata_summary"])

    def test_partition_summary_uses_float32_numpy_mean_and_handles_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            vectors = [
                [index / 10.0, (-1.0) ** index * index, index % 7]
                for index in range(257)
            ]
            records = [
                VectorRecord(f"doc{index}", vector, {"tenant": "a"})
                for index, vector in enumerate(vectors)
            ]

            manifest = BootstrapExecutor(
                partition_store,
                routing_store,
            ).create_partition("p0000", "g0", records)
            summary = partition_store.read_base_summary(manifest)
            expected = np.asarray(
                vectors,
                dtype=np.float32,
            ).mean(axis=0, dtype=np.float64)

            np.testing.assert_allclose(summary["centroid"], expected)
            self.assertEqual(summary["metadata_summary"], {"tenant": {"a": 257}})

            empty = BootstrapExecutor(
                partition_store,
                routing_store,
            ).create_partition("p0001", "g0")
            self.assertEqual(
                partition_store.read_base_summary(empty),
                {"count": 0, "centroid": None, "metadata_summary": {}},
            )

    def test_routing_updates_use_checkpoint_interval_layout_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, routing_store = self._system(tmpdir)
            BootstrapExecutor(partition_store, routing_store).create_partition(
                "p0000",
                "g0",
                [
                    VectorRecord("doc1", [1.0, 3.0], {"tenant": "a", "project": "x"}),
                    VectorRecord("doc2", [3.0, 1.0], {"tenant": "a", "project": "y"}),
                ],
            )

            keys = object_store.list("datasets/test/routing/partition_updates/after_000000")
            self.assertEqual(len(keys), 1)
            self.assertIn("update_cp000000_p0000_m000001_", keys[0])
            view = routing_store.effective_view()
            entry = view["p0000"]
            self.assertEqual(entry.count, 2)
            self.assertEqual(entry.summary_covered_delta_epoch, 0)
            self.assertEqual(entry.centroid, [2.0, 2.0])
            self.assertEqual(entry.metadata_summary["tenant"], {"a": 2})
            self.assertEqual(entry.metadata_summary["project"], {"x": 1, "y": 1})

    def test_query_uses_metadata_routing_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            bootstrap = BootstrapExecutor(partition_store, routing_store)
            bootstrap.create_partition("p0000", "g0", [
                VectorRecord("doc-a", [1.0, 0.0], {"tenant": "a"}),
            ])
            bootstrap.create_partition("p0001", "g0", [
                VectorRecord("doc-b", [0.0, 1.0], {"tenant": "b"}),
            ])

            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query(
                {"g0": [1.0, 0.0]},
                top_k=10,
                metadata_filter={"tenant": "b"},
            )

            self.assertEqual([hit.id for hit in hits], ["doc-b"])

    def test_query_uses_centroid_routing_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            bootstrap = BootstrapExecutor(partition_store, routing_store)
            bootstrap.create_partition("p0000", "g0", [VectorRecord("doc-a", [1.0, 0.0])])
            bootstrap.create_partition("p0001", "g0", [VectorRecord("doc-b", [0.0, 1.0])])

            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query({"g0": [1.0, 0.0]}, top_k=10, routing_top_k=1)

            self.assertEqual([hit.id for hit in hits], ["doc-a"])

    def test_routing_checkpoint_merges_current_interval_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, partition_store, routing_store = self._system(tmpdir)
            bootstrap = BootstrapExecutor(partition_store, routing_store)
            bootstrap.create_partition("p0000", "g0", [VectorRecord("doc-a", [1.0, 0.0])])
            bootstrap.create_partition("p0001", "g0", [VectorRecord("doc-b", [0.0, 1.0])])
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())

            self.assertTrue(routing_store.publish_merged_checkpoint())
            checkpoint, _ = routing_store.read_current_checkpoint()

            self.assertEqual(checkpoint.epoch, 1)
            self.assertEqual(set(checkpoint.entries), {"p0000", "p0001"})
            self.assertEqual(routing_store.list_pending_updates(1), [])
            self.assertEqual(set(routing_store.effective_view()), {"p0000", "p0001"})
            self.assertEqual(
                object_store.list("datasets/test/routing/partition_updates/after_000001"),
                [],
            )

            ingest.ingest([VectorRecord("doc-c", [0.5, 0.5])], candidate_partitions=["p0000"])
            self.assertTrue(CompactionExecutor(partition_store, routing_store).compact_partition("p0000").committed)
            keys = object_store.list("datasets/test/routing/partition_updates/after_000001")
            self.assertEqual(len(keys), 1)
            self.assertIn("update_cp000001_p0000_m000003_", keys[0])

    def test_compaction_routing_summary_covers_compacted_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])], candidate_partitions=["p0000"])
            ingest.ingest([VectorRecord("doc2", [0.0, 1.0])], candidate_partitions=["p0000"])

            result = CompactionExecutor(partition_store, routing_store).compact_partition("p0000")

            self.assertTrue(result.committed)
            entry = routing_store.effective_view()["p0000"]
            self.assertEqual(entry.count, 2)
            self.assertEqual(entry.summary_covered_delta_epoch, 2)
            self.assertEqual(entry.centroid, [0.5, 0.5])


    def test_generation_migration_switches_active_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([
                VectorRecord("doc1", [1.0, 0.0], {"tenant": "a"}, "source")
            ], candidate_partitions=["p0000"])

            migration = MigrationExecutor(partition_store, routing_store)
            result = migration.migrate_partition("p0000", "g1", SwapGenerationTransformer())

            self.assertTrue(result.committed)
            self.assertFalse(result.conflict)
            self.assertEqual(result.source_generation, "g0")
            self.assertEqual(result.target_generation, "g1")
            self.assertEqual(result.live_count, 1)
            phase_values = [
                result.timings.read_seconds,
                result.timings.transform_seconds,
                result.timings.index_build_seconds,
                result.timings.serialize_seconds,
                result.timings.write_seconds,
                result.timings.publish_seconds,
            ]
            self.assertTrue(all(value >= 0.0 for value in phase_values))
            self.assertLessEqual(sum(phase_values), result.timings.total_seconds)
            self.assertGreater(result.timings.total_seconds, 0.0)

            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.active_generation, "g1")
            self.assertEqual(manifest.latest_base_version, 2)
            self.assertEqual(manifest.latest_delta_epoch, 1)
            self.assertEqual(manifest.visible_deltas, [])
            self.assertIsNotNone(manifest.base)
            self.assertEqual(manifest.base.generation, "g1")
            self.assertTrue(manifest.base.version.startswith("b000002_"))
            self.assertIn("g0", manifest.retained_generations)
            self.assertEqual(manifest.retained_generations["g0"].generation, "g0")

            query = QueryExecutor(partition_store, routing_store, index)
            self.assertEqual(query.query({"g0": [1.0, 0.0]}, top_k=10), [])
            hits = query.query({"g1": [0.0, 1.0]}, top_k=10)
            self.assertEqual([hit.id for hit in hits], ["doc1"])
            entry = routing_store.effective_view()["p0000"]
            self.assertEqual(entry.active_generation, "g1")
            self.assertEqual(entry.summary_covered_delta_epoch, 1)
            self.assertEqual(entry.centroid, [0.0, 1.0])
            self.assertEqual(entry.metadata_summary["tenant"], {"a": 1})

    def test_mixed_generation_query_searches_matching_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            bootstrap = BootstrapExecutor(partition_store, routing_store)
            bootstrap.create_partition("p0000", "g0", [VectorRecord("doc-g0", [1.0, 0.0])])
            bootstrap.create_partition("p0001", "g0", [VectorRecord("doc-g1", [1.0, 0.0])])
            MigrationExecutor(partition_store, routing_store).migrate_partition(
                "p0001", "g1", SwapGenerationTransformer()
            )

            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query({"g0": [1.0, 0.0], "g1": [0.0, 1.0]}, top_k=10)

            self.assertEqual({hit.id for hit in hits}, {"doc-g0", "doc-g1"})
            self.assertEqual({hit.generation for hit in hits}, {"g0", "g1"})

    def test_generation_migration_skips_target_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            migration = MigrationExecutor(partition_store, routing_store)

            result = migration.migrate_partition("p0000", "g0", SwapGenerationTransformer())

            self.assertFalse(result.committed)
            self.assertTrue(result.skipped)
            self.assertEqual(result.reason, "already on target generation")
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.version, 1)
            self.assertEqual(manifest.active_generation, "g0")

    def test_generation_migration_skips_split_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            manifest = partition_store.create_partition("p0000", "g0")
            _, handle = partition_store.read_current_manifest("p0000")
            split_manifest = PartitionManifest(
                partition_id="p0000",
                version=manifest.version + 1,
                active_generation="g0",
                latest_base_version=manifest.latest_base_version,
                latest_delta_epoch=manifest.latest_delta_epoch,
                state="split",
                base=manifest.base,
                children=["p0000a", "p0000b"],
            )
            self.assertTrue(partition_store.publish_manifest(handle, split_manifest))
            migration = MigrationExecutor(partition_store, routing_store)

            result = migration.migrate_partition("p0000", "g1", SwapGenerationTransformer())

            self.assertFalse(result.committed)
            self.assertTrue(result.skipped)
            self.assertEqual(result.reason, "partition is not active")

    def test_generation_migration_reports_manifest_publish_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])], candidate_partitions=["p0000"])
            original_publish = partition_store.publish_manifest

            def racing_publish(handle, candidate, commit_id=None):
                current, fresh_handle = partition_store.read_current_manifest("p0000")
                competing = PartitionManifest(
                    partition_id="p0000",
                    version=current.version + 1,
                    active_generation=current.active_generation,
                    latest_base_version=current.latest_base_version,
                    latest_delta_epoch=current.latest_delta_epoch,
                    state=current.state,
                    base=current.base,
                    visible_deltas=current.visible_deltas,
                    retained_generations=current.retained_generations,
                    children=current.children,
                )
                self.assertTrue(original_publish(fresh_handle, competing, new_commit_id()))
                return original_publish(handle, candidate, commit_id)

            partition_store.publish_manifest = racing_publish
            migration = MigrationExecutor(partition_store, routing_store)

            result = migration.migrate_partition("p0000", "g1", SwapGenerationTransformer())

            self.assertFalse(result.committed)
            self.assertTrue(result.conflict)
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(manifest.active_generation, "g0")
            query = QueryExecutor(partition_store, routing_store, index)
            self.assertEqual(query.query({"g0": [1.0, 0.0]}, top_k=1)[0].id, "doc1")
            self.assertEqual(query.query({"g1": [0.0, 1.0]}, top_k=1), [])

    def test_generation_migration_rejects_invalid_transformer_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])], candidate_partitions=["p0000"])
            migration = MigrationExecutor(partition_store, routing_store)

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                migration.migrate_partition("p0000", "g1", EmptyVectorTransformer())

    def test_partition_split_materializes_live_state_and_forwards_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            policy = HashPartitionPolicy()
            ingest = IngestExecutor(partition_store, routing_store, policy)
            ingest.ingest([
                VectorRecord("doc1", [1.0, 0.0], text="old"),
                VectorRecord("doc2", [0.0, 1.0]),
                VectorRecord("doc3", [0.5, 0.5]),
            ], candidate_partitions=["p0000"])
            ingest.ingest([
                VectorRecord("doc1", [0.8, 0.2], text="new"),
            ], tombstones=["doc2"], candidate_partitions=["p0000"])

            result = SplitExecutor(partition_store, routing_store).split_partition(
                "p0000",
                ["p0000a", "p0000b"],
                policy,
            )

            self.assertTrue(result.committed)
            self.assertFalse(result.conflict)
            self.assertEqual(result.live_count, 2)
            self.assertEqual(sum(result.child_counts.values()), 2)
            source, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(source.state, "split")
            self.assertEqual(source.children, ["p0000a", "p0000b"])
            self.assertEqual(
                partition_store.list_partitions(),
                ["p0000a", "p0000b"],
            )
            for child_id in source.children:
                child, _ = partition_store.read_current_manifest(child_id)
                self.assertEqual(child.parent_partition_id, "p0000")
                self.assertEqual(child.active_generation, "g0")
                self.assertEqual(child.visible_deltas, [])

            view = routing_store.effective_view()
            self.assertEqual(set(view), {"p0000a", "p0000b"})
            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query(
                {"g0": [1.0, 0.0]},
                top_k=10,
                candidate_partitions=["p0000"],
            )
            self.assertEqual({hit.id for hit in hits}, {"doc1", "doc3"})
            self.assertEqual(
                next(hit for hit in hits if hit.id == "doc1").text,
                "new",
            )

    def test_ingest_redirects_split_source_to_child_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            policy = HashPartitionPolicy()
            split = SplitExecutor(partition_store, routing_store)
            self.assertTrue(
                split.split_partition(
                    "p0000",
                    ["p0000a", "p0000b"],
                    policy,
                ).committed
            )
            record = VectorRecord("new-doc", [1.0, 0.0])
            expected_child = policy.assign(record, ["p0000a", "p0000b"])

            result = IngestExecutor(
                partition_store,
                routing_store,
                policy,
            ).ingest([record], candidate_partitions=["p0000"])

            self.assertEqual(result.committed_partitions, [expected_child])
            child, _ = partition_store.read_current_manifest(expected_child)
            self.assertEqual(child.latest_delta_epoch, 1)
            query = QueryExecutor(partition_store, routing_store, index)
            hits = query.query(
                {"g0": [1.0, 0.0]},
                top_k=10,
                candidate_partitions=["p0000"],
            )
            self.assertEqual([hit.id for hit in hits], ["new-doc"])

    def test_split_conflict_keeps_candidate_children_out_of_leaf_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
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
                    retained_generations=current.retained_generations,
                    parent_partition_id=current.parent_partition_id,
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

            self.assertFalse(result.committed)
            self.assertTrue(result.conflict)
            source, _ = partition_store.read_current_manifest("p0000")
            self.assertEqual(source.state, "active")
            self.assertEqual(partition_store.list_partitions(), ["p0000"])
            query = QueryExecutor(partition_store, routing_store, index)
            self.assertEqual(
                [hit.id for hit in query.query({"g0": [1.0, 0.0]}, top_k=10)],
                ["doc1"],
            )

    def test_split_checkpoint_tombstone_rejects_stale_source_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
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
            result = SplitExecutor(partition_store, routing_store).split_partition(
                "p0000",
                ["p0000a", "p0000b"],
                HashPartitionPolicy(),
            )
            self.assertTrue(result.committed)
            self.assertTrue(routing_store.publish_merged_checkpoint())
            checkpoint, _ = routing_store.read_current_checkpoint()
            self.assertEqual(checkpoint.entries["p0000"].state, "split")

            routing_store.write_partition_update(
                RoutingEntry(
                    partition_id="p0000",
                    manifest_version=result.source_version,
                    active_generation="g0",
                    count=1,
                ),
                commit_id="stale-after-split",
            )

            self.assertEqual(
                set(routing_store.effective_view()),
                {"p0000a", "p0000b"},
            )

    def test_partition_split_validates_children_and_skips_forwarder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            split = SplitExecutor(partition_store, routing_store)
            policy = HashPartitionPolicy()

            with self.assertRaisesRegex(ValueError, "must be unique"):
                split.split_partition("p0000", ["child", "child"], policy)

            partition_store.create_partition("taken", "g0")
            with self.assertRaisesRegex(ValueError, "already exists"):
                split.split_partition("p0000", ["taken", "other"], policy)

            self.assertTrue(
                split.split_partition("p0000", ["p0000a", "p0000b"], policy).committed
            )
            result = split.split_partition("p0000", ["unused-a", "unused-b"], policy)
            self.assertFalse(result.committed)
            self.assertTrue(result.skipped)
            self.assertEqual(result.reason, "partition is not active")


    def test_routing_update_writes_once_when_checkpoint_epoch_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, object_store, _, routing_store = self._system(tmpdir)
            entry = RoutingEntry(
                partition_id="p0000",
                manifest_version=2,
                active_generation="g0",
                count=1,
            )

            routing_store.write_partition_update(entry, commit_id="stable")

            self.assertEqual(
                len(object_store.list("datasets/test/routing/partition_updates/after_000000")),
                1,
            )
            self.assertEqual(
                object_store.list("datasets/test/routing/partition_updates/after_000001"),
                [],
            )

    def test_late_routing_update_is_reemitted_under_latest_checkpoint_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store = LocalObjectStore(tmpdir)
            routing_store = RacingRoutingStore(
                object_store,
                "test",
                RoutingCheckpoint(epoch=1),
            )
            entry = RoutingEntry(
                partition_id="p0000",
                manifest_version=2,
                active_generation="g0",
                count=1,
            )

            routing_store.write_partition_update(entry, commit_id="late")

            self.assertEqual(
                len(object_store.list("datasets/test/routing/partition_updates/after_000000")),
                1,
            )
            latest_keys = object_store.list("datasets/test/routing/partition_updates/after_000001")
            self.assertEqual(len(latest_keys), 1)
            self.assertIn("update_cp000001_p0000_m000002_", latest_keys[0])

    def test_obsolete_late_routing_update_is_not_reemitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store = LocalObjectStore(tmpdir)
            newer_entry = RoutingEntry(
                partition_id="p0000",
                manifest_version=3,
                active_generation="g0",
                count=2,
            )
            routing_store = RacingRoutingStore(
                object_store,
                "test",
                RoutingCheckpoint(epoch=1, entries={"p0000": newer_entry}),
            )
            stale_entry = RoutingEntry(
                partition_id="p0000",
                manifest_version=2,
                active_generation="g0",
                count=1,
            )

            routing_store.write_partition_update(stale_entry, commit_id="stale")

            self.assertEqual(
                len(object_store.list("datasets/test/routing/partition_updates/after_000000")),
                1,
            )
            self.assertEqual(
                object_store.list("datasets/test/routing/partition_updates/after_000001"),
                [],
            )

    def test_tombstone_filters_deleted_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, _, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
            ingest.ingest([VectorRecord("doc1", [1.0, 0.0])])
            ingest.ingest([], tombstones=["doc1"])

            query = QueryExecutor(partition_store, routing_store, index)
            self.assertEqual(query.query({"g0": [1.0, 0.0]}, top_k=10), [])


if __name__ == "__main__":
    unittest.main()
