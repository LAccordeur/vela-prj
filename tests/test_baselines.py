"""Tests for offline full-dataset partitioning baselines."""

from __future__ import annotations

import tempfile
import unittest

from vela.execution.baseline import BaselineExecutor
from vela.execution.query import QueryExecutor
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore


class BaselineExecutorTest(unittest.TestCase):
    """Covers FAISS clustering and contiguous block publication."""

    def _system(self, tmpdir: str):
        object_store = LocalObjectStore(tmpdir)
        index = FlatIndex()
        partition_store = PartitionStore(object_store, index, dataset_id="test")
        routing_store = RoutingStore(object_store, dataset_id="test")
        executor = BaselineExecutor(partition_store, routing_store)
        return index, partition_store, routing_store, executor

    def test_clustering_build_groups_separated_vectors_and_is_queryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index, partition_store, routing_store, executor = self._system(tmpdir)
            records = [
                VectorRecord("left-1", [-10.0, 0.0]),
                VectorRecord("left-2", [-9.0, 0.0]),
                VectorRecord("right-1", [9.0, 0.0]),
                VectorRecord("right-2", [10.0, 0.0]),
            ]

            result = executor.build_clustering(
                records,
                partition_count=2,
                generation="g0",
                seed=7,
            )

            self.assertEqual(result.partition_ids, ["p0000", "p0001"])
            self.assertEqual(sum(result.partition_counts.values()), 4)
            groups = []
            for partition_id in result.partition_ids:
                manifest, _ = partition_store.read_current_manifest(partition_id)
                groups.append({
                    record.id
                    for record in partition_store.materialize_live_records(manifest)
                })
                self.assertEqual(manifest.visible_deltas, [])
            self.assertCountEqual(
                groups,
                [
                    {"left-1", "left-2"},
                    {"right-1", "right-2"},
                ],
            )
            hits = QueryExecutor(partition_store, routing_store, index).query(
                {"g0": [1.0, 0.0]},
                top_k=1,
            )
            self.assertIn(hits[0].id, {"right-1", "right-2"})

    def test_block_build_preserves_order_in_balanced_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, routing_store, executor = self._system(tmpdir)
            records = [
                VectorRecord(f"doc-{index}", [float(index), 1.0])
                for index in range(7)
            ]

            result = executor.build_blocks(
                records,
                partition_count=3,
                generation="g0",
                partition_prefix="block-",
            )

            self.assertEqual(
                result.partition_counts,
                {"block-0000": 3, "block-0001": 2, "block-0002": 2},
            )
            actual = []
            for partition_id in result.partition_ids:
                manifest, _ = partition_store.read_current_manifest(partition_id)
                actual.append([
                    record.id
                    for record in partition_store.materialize_live_records(manifest)
                ])
            self.assertEqual(
                actual,
                [
                    ["doc-0", "doc-1", "doc-2"],
                    ["doc-3", "doc-4"],
                    ["doc-5", "doc-6"],
                ],
            )
            self.assertEqual(len(routing_store.effective_view()), 3)

    def test_build_rejects_non_vector_fields_and_existing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, partition_store, _, executor = self._system(tmpdir)
            with self.assertRaisesRegex(ValueError, "vector data only"):
                executor.build_blocks(
                    [VectorRecord("doc", [1.0], metadata={"kind": "test"})],
                    partition_count=1,
                    generation="g0",
                )

            partition_store.create_partition("existing", "g0")
            with self.assertRaisesRegex(ValueError, "without partitions"):
                executor.build_blocks(
                    [VectorRecord("doc", [1.0])],
                    partition_count=1,
                    generation="g0",
                )

    def test_build_validates_dataset_shape_and_partition_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, executor = self._system(tmpdir)
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                executor.build_blocks([], 1, "g0")
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                executor.build_blocks(
                    [VectorRecord("doc", [1.0])],
                    2,
                    "g0",
                )
            with self.assertRaisesRegex(ValueError, "same dimension"):
                executor.build_clustering(
                    [
                        VectorRecord("a", [1.0]),
                        VectorRecord("b", [1.0, 2.0]),
                    ],
                    2,
                    "g0",
                )


if __name__ == "__main__":
    unittest.main()
