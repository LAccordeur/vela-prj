"""Tests for querying manifest-referenced persistent ANN indexes."""

from __future__ import annotations

import tempfile
from threading import Lock
import time
import unittest

from vela.execution.ingest import IngestExecutor
from vela.execution.query import QueryExecutor
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore, ReferencedObjectMissing
from vela.storage.routing_store import RoutingStore


class SearchOnlyIndex:
    """Flat index wrapper that fails if query tries to rebuild an index."""

    def __init__(self) -> None:
        self.delegate = FlatIndex()
        self.search_limits: list[int] = []

    def build(self, records: list[VectorRecord]) -> bytes:
        raise AssertionError("query must not rebuild a persistent index")

    def load(self, index_blob: bytes):
        return SearchOnlyLoadedIndex(
            self.delegate.load(index_blob),
            self.search_limits,
        )


class SearchOnlyLoadedIndex:
    """Loaded-index wrapper that records adaptive search limits."""

    def __init__(self, delegate, search_limits: list[int]) -> None:
        self.delegate = delegate
        self.search_limits = search_limits
        self.count = delegate.count
        self.dimension = delegate.dimension
        self.metric = delegate.metric

    def search(self, query: list[float], top_k: int):
        self.search_limits.append(top_k)
        return self.delegate.search(query, top_k)


class TrackingLocalObjectStore(LocalObjectStore):
    """Local store that records concurrent object GET operations."""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self._tracking_lock = Lock()
        self.active_gets = 0
        self.max_active_gets = 0

    def get(self, key: str) -> bytes:
        with self._tracking_lock:
            self.active_gets += 1
            self.max_active_gets = max(
                self.max_active_gets,
                self.active_gets,
            )
        try:
            time.sleep(0.01)
            return super().get(key)
        finally:
            with self._tracking_lock:
                self.active_gets -= 1

    def reset_tracking(self) -> None:
        with self._tracking_lock:
            self.active_gets = 0
            self.max_active_gets = 0


class PersistentQueryTest(unittest.TestCase):
    """Covers persistent base/delta search and logical visibility."""

    def _system(self, tmpdir: str):
        object_store = LocalObjectStore(tmpdir)
        partition_store = PartitionStore(
            object_store,
            FlatIndex(),
            dataset_id="test",
        )
        routing_store = RoutingStore(object_store, dataset_id="test")
        return object_store, partition_store, routing_store

    def test_partition_and_object_reads_are_bounded_and_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store = TrackingLocalObjectStore(tmpdir)
            partition_store = PartitionStore(
                object_store,
                FlatIndex(),
                dataset_id="parallel",
            )
            routing_store = RoutingStore(object_store, dataset_id="parallel")
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("left", [0.0, 0.0])],
            )
            partition_store.create_partition(
                "p0001",
                "g0",
                [VectorRecord("right", [1.0, 0.0])],
            )

            with QueryExecutor(
                partition_store,
                routing_store,
                FlatIndex(),
            ) as serial:
                expected = serial.query(
                    {"g0": [0.0, 0.0]},
                    top_k=2,
                    candidate_partitions=["p0000", "p0001"],
                )

            object_store.reset_tracking()
            with QueryExecutor(
                partition_store,
                routing_store,
                FlatIndex(),
                partition_workers=2,
                object_read_workers=2,
            ) as parallel:
                actual, profile = parallel.query_profiled(
                    {"g0": [0.0, 0.0]},
                    top_k=2,
                    candidate_partitions=["p0000", "p0001"],
                )

            self.assertEqual(
                [hit.id for hit in actual],
                [hit.id for hit in expected],
            )
            self.assertGreater(object_store.max_active_gets, 1)
            self.assertLessEqual(object_store.max_active_gets, 2)
            self.assertEqual(profile.phase_accounting, "worker-seconds")
            with self.assertRaisesRegex(ValueError, "workers"):
                QueryExecutor(
                    partition_store,
                    routing_store,
                    FlatIndex(),
                    object_read_workers=0,
                )



    def test_one_partition_reads_multiple_segments_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store = TrackingLocalObjectStore(tmpdir)
            partition_store = PartitionStore(
                object_store,
                FlatIndex(),
                dataset_id="segments",
            )
            routing_store = RoutingStore(object_store, dataset_id="segments")
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("base", [0.0, 0.0])],
            )
            ingest = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            )
            for ordinal in range(2):
                result = ingest.ingest(
                    [VectorRecord(f"delta-{ordinal}", [float(ordinal), 0.0])],
                    candidate_partitions=["p0000"],
                )
                self.assertFalse(result.conflicts)

            object_store.reset_tracking()
            with QueryExecutor(
                partition_store,
                routing_store,
                FlatIndex(),
                partition_workers=1,
                object_read_workers=2,
            ) as executor:
                hits = executor.query(
                    {"g0": [0.0, 0.0]},
                    top_k=3,
                    candidate_partitions=["p0000"],
                )

            self.assertEqual(len(hits), 3)
            self.assertGreater(object_store.max_active_gets, 1)
            self.assertLessEqual(object_store.max_active_gets, 2)

    def test_query_searches_persistent_base_and_delta_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [
                    VectorRecord("base-live", [0.8, 0.2]),
                    VectorRecord("updated", [1.0, 0.0], text="old"),
                ],
            )
            ingest = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            )
            ingest.ingest(
                [
                    VectorRecord("updated", [-1.0, 0.0], text="new"),
                    VectorRecord("delta-live", [0.9, 0.1]),
                ],
                candidate_partitions=["p0000"],
            )
            manifest, _ = partition_store.read_current_manifest("p0000")
            object_store.delete(manifest.base.vectors_key)
            for delta in manifest.visible_deltas:
                object_store.delete(delta.vectors_key)
            query_index = SearchOnlyIndex()

            executor = QueryExecutor(
                partition_store,
                routing_store,
                query_index,
            )
            hits = executor.query(
                {"g0": [1.0, 0.0]},
                top_k=3,
                candidate_partitions=["p0000"],
            )
            profiled_hits, profile = executor.query_profiled(
                {"g0": [1.0, 0.0]},
                top_k=3,
                candidate_partitions=["p0000"],
            )

            self.assertEqual(
                [hit.id for hit in hits],
                ["delta-live", "base-live", "updated"],
            )
            self.assertEqual(hits[-1].text, "new")
            self.assertTrue(query_index.search_limits)
            self.assertEqual(
                [hit.id for hit in profiled_hits],
                [hit.id for hit in hits],
            )
            self.assertEqual(profile.selected_partition_count, 1)
            self.assertEqual(profile.loaded_segment_count, 2)
            self.assertGreaterEqual(
                profile.total_seconds,
                sum(profile.phase_seconds().values()),
            )
            self.assertEqual(profile.routing_read_seconds, 0.0)
            self.assertGreater(profile.ids_read_seconds, 0.0)
            self.assertGreater(profile.index_read_seconds, 0.0)
            self.assertGreater(profile.index_deserialize_seconds, 0.0)
            self.assertGreater(profile.ann_search_seconds, 0.0)

    def test_query_expands_local_top_k_past_stale_high_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store = self._system(tmpdir)
            stale_ids = [f"stale-{idx}" for idx in range(5)]
            partition_store.create_partition(
                "p0000",
                "g0",
                [
                    *[
                        VectorRecord(record_id, [1.0, 0.0])
                        for record_id in stale_ids
                    ],
                    VectorRecord("live", [0.0, 1.0]),
                ],
            )
            IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [
                    VectorRecord(record_id, [-1.0, 0.0])
                    for record_id in stale_ids
                ],
                candidate_partitions=["p0000"],
            )
            query_index = SearchOnlyIndex()

            hits = QueryExecutor(
                partition_store,
                routing_store,
                query_index,
            ).query(
                {"g0": [1.0, 0.0]},
                top_k=1,
                candidate_partitions=["p0000"],
            )

            self.assertEqual([hit.id for hit in hits], ["live"])
            self.assertGreater(max(query_index.search_limits), 2)

    def test_empty_tombstones_omit_object_and_query_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("p0000", "g0")
            IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [VectorRecord("doc1", [1.0, 0.0])],
                candidate_partitions=["p0000"],
            )
            manifest, _ = partition_store.read_current_manifest("p0000")
            delta = manifest.visible_deltas[0]

            self.assertIsNone(delta.tombstones_key)
            self.assertFalse(
                any(
                    key.endswith("/tombstones.json")
                    for key in object_store.list(
                        "datasets/test/partitions/p0000/delta"
                    )
                )
            )
            hits = QueryExecutor(
                partition_store,
                routing_store,
                SearchOnlyIndex(),
            ).query(
                {"g0": [1.0, 0.0]},
                top_k=1,
                candidate_partitions=["p0000"],
            )
            self.assertEqual([hit.id for hit in hits], ["doc1"])

    def test_real_and_missing_tombstone_objects_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store, partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition(
                "p0000",
                "g0",
                [VectorRecord("doc1", [1.0, 0.0])],
            )
            IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [],
                tombstones=["doc1"],
                candidate_partitions=["p0000"],
            )
            manifest, _ = partition_store.read_current_manifest("p0000")
            tombstones_key = manifest.visible_deltas[-1].tombstones_key
            self.assertIsNotNone(tombstones_key)
            query = QueryExecutor(
                partition_store,
                routing_store,
                SearchOnlyIndex(),
                max_missing_retries=0,
            )
            self.assertEqual(
                query.query(
                    {"g0": [1.0, 0.0]},
                    top_k=1,
                    candidate_partitions=["p0000"],
                ),
                [],
            )

            object_store.delete(tombstones_key or "")
            with self.assertRaises(ReferencedObjectMissing):
                query.query(
                    {"g0": [1.0, 0.0]},
                    top_k=1,
                    candidate_partitions=["p0000"],
                )


if __name__ == "__main__":
    unittest.main()
