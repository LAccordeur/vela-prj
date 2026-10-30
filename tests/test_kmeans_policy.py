"""Tests for K-Means batch partition placement."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import faiss

from vela.execution.bootstrap import BootstrapExecutor
from vela.execution.ingest import IngestExecutor
from vela.execution.split import SplitExecutor
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.partition.kmeans_policy import KMeansPartitionPolicy
from vela.partition.policy import PartitionTarget
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore


class KMeansPartitionPolicyTest(unittest.TestCase):
    """Covers batch clustering and ingest integration."""

    def _system(self, tmpdir: str):
        object_store = LocalObjectStore(tmpdir)
        partition_store = PartitionStore(
            object_store,
            FlatIndex(),
            dataset_id="test",
        )
        routing_store = RoutingStore(object_store, dataset_id="test")
        return partition_store, routing_store

    def test_assign_batch_maps_configuration_to_faiss(self) -> None:
        policy = KMeansPartitionPolicy(
            n_clusters=2,
            random_state=5,
            max_iter=7,
            n_init=3,
        )
        records = [
            VectorRecord("left", [-1.0, 0.0]),
            VectorRecord("right", [1.0, 0.0]),
        ]
        targets = [PartitionTarget("p0"), PartitionTarget("p1")]

        with patch(
            "vela.partition.kmeans_policy.faiss.Kmeans",
            wraps=faiss.Kmeans,
        ) as kmeans:
            policy.assign_batch(records, targets)

        kmeans.assert_called_once_with(
            2, 2, niter=7, nredo=3, seed=5, verbose=False, gpu=False,
        )

    def test_assign_batch_places_clusters_near_partition_centroids(self) -> None:
        policy = KMeansPartitionPolicy(n_clusters=2, random_state=7)
        records = [
            VectorRecord("left-1", [-10.0, 0.0]),
            VectorRecord("left-2", [-9.0, 0.0]),
            VectorRecord("right-1", [9.0, 0.0]),
            VectorRecord("right-2", [10.0, 0.0]),
        ]
        targets = [
            PartitionTarget("p-left", [-10.0, 0.0]),
            PartitionTarget("p-right", [10.0, 0.0]),
        ]

        assignments = policy.assign_batch(records, targets)

        self.assertEqual(assignments[:2], ["p-left", "p-left"])
        self.assertEqual(assignments[2:], ["p-right", "p-right"])
        self.assertEqual(assignments, policy.assign_batch(records, targets))

    def test_assign_batch_bootstraps_partitions_without_centroids(self) -> None:
        policy = KMeansPartitionPolicy(n_clusters=2, random_state=3)
        assignments = policy.assign_batch(
            [
                VectorRecord("left-1", [-5.0, 0.0]),
                VectorRecord("left-2", [-4.0, 0.0]),
                VectorRecord("right-1", [4.0, 0.0]),
                VectorRecord("right-2", [5.0, 0.0]),
            ],
            [PartitionTarget("p0"), PartitionTarget("p1")],
        )

        self.assertEqual(set(assignments), {"p0", "p1"})
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])

    def test_ingest_uses_routing_centroids_for_batch_placement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            partition_store, routing_store = self._system(tmpdir)
            bootstrap = BootstrapExecutor(partition_store, routing_store)
            bootstrap.create_partition(
                "p-left",
                "g0",
                [VectorRecord("seed-left", [-10.0, 0.0])],
            )
            bootstrap.create_partition(
                "p-right",
                "g0",
                [VectorRecord("seed-right", [10.0, 0.0])],
            )

            result = IngestExecutor(
                partition_store,
                routing_store,
                KMeansPartitionPolicy(n_clusters=2, random_state=11),
            ).ingest(
                [
                    VectorRecord("left-1", [-9.5, 0.0]),
                    VectorRecord("left-2", [-9.0, 0.0]),
                    VectorRecord("right-1", [9.0, 0.0]),
                    VectorRecord("right-2", [9.5, 0.0]),
                ]
            )

            self.assertEqual(set(result.committed_partitions), {"p-left", "p-right"})
            self.assertEqual(result.conflicts, [])
            left, _ = partition_store.read_current_manifest("p-left")
            right, _ = partition_store.read_current_manifest("p-right")
            self.assertEqual(
                {record.id for record in partition_store.materialize_live_records(left)},
                {"seed-left", "left-1", "left-2"},
            )
            self.assertEqual(
                {record.id for record in partition_store.materialize_live_records(right)},
                {"seed-right", "right-1", "right-2"},
            )

    def test_kmeans_batch_is_reclustered_after_split_forwarding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            partition_store, routing_store = self._system(tmpdir)
            partition_store.create_partition("source", "g0")
            seed = IngestExecutor(
                partition_store,
                routing_store,
                HashPartitionPolicy(),
            )
            seed.ingest(
                [
                    VectorRecord("seed-left", [-10.0, 0.0]),
                    VectorRecord("seed-right", [10.0, 0.0]),
                ],
                candidate_partitions=["source"],
            )
            policy = KMeansPartitionPolicy(n_clusters=2, random_state=5)
            split = SplitExecutor(partition_store, routing_store).split_partition(
                "source",
                ["child-a", "child-b"],
                policy,
            )
            self.assertTrue(split.committed)

            view = routing_store.effective_view()
            incoming = [
                VectorRecord(
                    f"near-{child_id}",
                    list(view[child_id].centroid or []),
                )
                for child_id in sorted(view)
            ]
            result = IngestExecutor(
                partition_store,
                routing_store,
                policy,
            ).ingest(incoming, candidate_partitions=["source"])

            self.assertEqual(set(result.committed_partitions), {"child-a", "child-b"})
            self.assertEqual(result.conflicts, [])

    def test_incompatible_centroid_is_never_used_as_bootstrap_target(self) -> None:
        policy = KMeansPartitionPolicy(n_clusters=2, random_state=13)

        assignments = policy.assign_batch(
            [
                VectorRecord("left", [-1.0, 0.0]),
                VectorRecord("right", [1.0, 0.0]),
            ],
            [
                PartitionTarget("empty"),
                PartitionTarget("wrong-dimension", [0.0]),
            ],
        )

        self.assertEqual(assignments, ["empty", "empty"])

    def test_assign_batch_rejects_invalid_records(self) -> None:
        policy = KMeansPartitionPolicy(n_clusters=2)
        targets = [PartitionTarget("p0", [0.0, 0.0])]

        with self.assertRaisesRegex(ValueError, "same dimension"):
            policy.assign_batch(
                [VectorRecord("a", [0.0]), VectorRecord("b", [0.0, 1.0])],
                targets,
            )
        with self.assertRaisesRegex(ValueError, "duplicate record id"):
            policy.assign_batch(
                [VectorRecord("same", [0.0, 0.0]), VectorRecord("same", [1.0, 1.0])],
                targets,
            )
        with self.assertRaisesRegex(ValueError, "compatible centroid"):
            policy.assign_batch(
                [VectorRecord("a", [0.0, 0.0])],
                [PartitionTarget("wrong-dimension", [0.0])],
            )


if __name__ == "__main__":
    unittest.main()
