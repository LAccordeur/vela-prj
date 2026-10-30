"""Tests for FlatIndex metrics and metric-aligned routing."""

from __future__ import annotations

import tempfile
import unittest

from vela.execution.bootstrap import BootstrapExecutor
from vela.execution.query import QueryExecutor
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore


class FlatIndexMetricTest(unittest.TestCase):
    """Covers L2 defaults, cosine compatibility, and centroid routing."""

    def test_l2_is_default_and_scores_are_negative_squared_distance(self) -> None:
        index = FlatIndex()
        artifact = index.build_profiled([
            VectorRecord("far-same-direction", [10.0, 0.0]),
            VectorRecord("near", [1.0, 1.0]),
        ])
        blob = artifact.blob

        hits = index.load(blob).search([1.0, 0.0], top_k=2)

        self.assertEqual([hit.position for hit in hits], [1, 0])
        self.assertGreaterEqual(artifact.build_seconds, 0.0)
        self.assertGreaterEqual(artifact.serialize_seconds, 0.0)
        self.assertEqual([hit.score for hit in hits], [-1.0, -81.0])

    def test_search_honors_cosine_metric_persisted_in_blob(self) -> None:
        cosine = FlatIndex(metric="cosine")
        blob = cosine.build([
            VectorRecord("same-direction", [10.0, 0.0]),
            VectorRecord("near-l2", [1.0, 1.0]),
        ])

        hits = FlatIndex().load(blob).search([1.0, 0.0], top_k=2)

        self.assertEqual([hit.position for hit in hits], [0, 1])
        self.assertAlmostEqual(hits[0].score, 1.0)

    def test_default_l2_routing_selects_nearest_partition_centroid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store = LocalObjectStore(tmpdir)
            index = FlatIndex()
            partition_store = PartitionStore(object_store, index, dataset_id="test")
            routing_store = RoutingStore(object_store, dataset_id="test")
            bootstrap = BootstrapExecutor(partition_store, routing_store)
            bootstrap.create_partition(
                "far",
                "g0",
                [VectorRecord("far-doc", [10.0, 0.0])],
            )
            bootstrap.create_partition(
                "near",
                "g0",
                [VectorRecord("near-doc", [1.0, 1.0])],
            )

            hits = QueryExecutor(
                partition_store,
                routing_store,
                index,
            ).query(
                {"g0": [1.0, 0.0]},
                top_k=1,
                routing_top_k=1,
            )

            self.assertEqual([hit.id for hit in hits], ["near-doc"])

    def test_invalid_metric_and_dimension_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            FlatIndex(metric="dot-product")
        blob = FlatIndex().build([VectorRecord("doc", [1.0, 2.0])])
        with self.assertRaisesRegex(ValueError, "same dimension"):
            FlatIndex().load(blob).search([1.0], top_k=1)


if __name__ == "__main__":
    unittest.main()
