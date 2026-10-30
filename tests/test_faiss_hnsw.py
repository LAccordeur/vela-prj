"""Tests for the persisted FAISS HNSW backend."""

from __future__ import annotations

import tempfile
import unittest

from vela.execution.bootstrap import BootstrapExecutor
from vela.execution.gc import GarbageCollector
from vela.execution.ingest import IngestExecutor
from vela.execution.query import QueryExecutor
from vela.index.faiss_hnsw import FaissHNSWIndex
from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore


class FaissHNSWIndexTest(unittest.TestCase):
    """Covers HNSW serialization, metrics, and manifest query integration."""

    def test_l2_round_trip_returns_positions_and_negative_distances(self) -> None:
        backend = FaissHNSWIndex(m=8, ef_construction=40, ef_search=20)
        artifact = backend.build_profiled([
            VectorRecord("far", [10.0, 0.0]),
            VectorRecord("near", [1.0, 1.0]),
        ])
        blob = artifact.blob

        loaded = backend.load(blob)
        hits = loaded.search([1.0, 0.0], top_k=2)

        self.assertEqual(loaded.count, 2)
        self.assertEqual(loaded.dimension, 2)
        self.assertEqual(loaded.metric, "l2")
        self.assertEqual([hit.position for hit in hits], [1, 0])
        self.assertAlmostEqual(hits[0].score, -1.0, places=5)
        self.assertAlmostEqual(hits[1].score, -81.0, places=5)
        self.assertNotIn(b"near", blob)
        self.assertGreaterEqual(artifact.build_seconds, 0.0)
        self.assertGreaterEqual(artifact.serialize_seconds, 0.0)

    def test_cosine_and_empty_indexes_round_trip(self) -> None:
        cosine = FaissHNSWIndex(
            metric="cosine",
            m=8,
            ef_construction=40,
            ef_search=20,
        )
        loaded = cosine.load(cosine.build([
            VectorRecord("same-direction", [10.0, 0.0]),
            VectorRecord("diagonal", [1.0, 1.0]),
        ]))
        hits = loaded.search([1.0, 0.0], top_k=2)

        self.assertEqual([hit.position for hit in hits], [0, 1])
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)

        empty = cosine.load(cosine.build([]))
        self.assertEqual(empty.count, 0)
        self.assertEqual(empty.dimension, 0)
        self.assertEqual(empty.search([1.0], top_k=1), [])

    def test_corrupt_envelope_and_payload_are_rejected(self) -> None:
        backend = FaissHNSWIndex(m=8)
        blob = backend.build([VectorRecord("doc", [1.0, 2.0])])

        with self.assertRaisesRegex(ValueError, "magic"):
            backend.load(b"BROKEN!!" + blob[8:])
        corrupted = bytearray(blob)
        corrupted[-1] ^= 0x01
        with self.assertRaisesRegex(ValueError, "checksum"):
            backend.load(bytes(corrupted))
        with self.assertRaisesRegex(ValueError, "same dimension"):
            backend.load(blob).search([1.0], top_k=1)

    def test_partition_query_uses_faiss_base_and_delta_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store = LocalObjectStore(tmpdir)
            backend = FaissHNSWIndex(
                m=8,
                ef_construction=40,
                ef_search=32,
            )
            partition_store = PartitionStore(
                object_store,
                backend,
                dataset_id="test",
            )
            routing_store = RoutingStore(object_store, dataset_id="test")
            BootstrapExecutor(
                partition_store,
                routing_store,
            ).create_partition(
                "p0000",
                "g0",
                [
                    VectorRecord("base-live", [0.0, 0.0]),
                    VectorRecord("deleted", [0.05, 0.0]),
                    VectorRecord("updated", [10.0, 10.0]),
                ],
            )
            IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            ).ingest(
                [
                    VectorRecord("updated", [0.1, 0.0]),
                    VectorRecord("delta-live", [0.2, 0.0]),
                ],
                tombstones=["deleted"],
                candidate_partitions=["p0000"],
            )
            manifest, _ = partition_store.read_current_manifest("p0000")
            self.assertTrue(manifest.base.index_key.endswith("/index.faiss"))
            self.assertTrue(
                manifest.visible_deltas[0].mini_index_key.endswith(
                    "/mini_index.faiss"
                )
            )

            object_store.delete(manifest.base.vectors_key)
            for delta in manifest.visible_deltas:
                object_store.delete(delta.vectors_key)

            hits = QueryExecutor(
                partition_store,
                routing_store,
                backend,
            ).query(
                {"g0": [0.0, 0.0]},
                top_k=3,
                candidate_partitions=["p0000"],
            )

            self.assertEqual(
                [hit.id for hit in hits],
                ["base-live", "updated", "delta-live"],
            )
            report = GarbageCollector(
                object_store,
                dataset_id="test",
            ).collect()
            self.assertIn(manifest.base.index_key, report.reachable_keys)
            self.assertIn(
                manifest.visible_deltas[0].mini_index_key,
                report.reachable_keys,
            )


if __name__ == "__main__":
    unittest.main()
