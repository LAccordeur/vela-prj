"""Tests for full-rebuild and partition-level evolution benchmarks."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.sift_evolution_benchmark import (
    EvolutionBenchmarkConfig,
    _faiss_threads_per_worker,
    _largest_partitions_first,
    _publish_collection_snapshot,
    run_evolution_benchmark,
)
from vela.storage.vector_codec import decode_float32_vectors
from vela.execution.migration import MigrationExecutor
from benchmarks.sift_evolution_benchmark import IdentityTransformer
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.utils.json_codec import from_json_bytes


class SiftEvolutionBenchmarkTest(unittest.TestCase):
    """Covers both global rebuild backends and incremental migration."""

    def test_staging_and_global_publication_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalObjectStore(tmpdir)
            partitions = PartitionStore(store, FlatIndex(), "test")
            migration = MigrationExecutor(partitions, RoutingStore(store, "test"))
            details = []
            pointer = "datasets/test/collection/generation.current"
            for partition_id in ("p0", "p1"):
                partitions.create_partition(partition_id, "g0", [VectorRecord(id=partition_id, vector=[1.0, 2.0])])
                result = migration.migrate_partition(partition_id, "g1", IdentityTransformer(), prepare_only=True)
                self.assertFalse(result.committed)
                self.assertIsNotNone(result.prepared_manifest_key)
                self.assertEqual(partitions.read_current_manifest(partition_id)[0].active_generation, "g0")
                self.assertIsNone(store.read_pointer(pointer).value)
                details.append({"partition_id": partition_id, "source_version": result.source_version,
                                "prepared_manifest_key": result.prepared_manifest_key,
                                "prepared_routing_entry": asdict(result.prepared_routing_entry)})
            with self.assertRaisesRegex(ValueError, "every partition"):
                _publish_collection_snapshot(partitions, ["p0", "p1"], details[:1])
            self.assertIsNone(store.read_pointer(pointer).value)
            key = _publish_collection_snapshot(partitions, ["p0", "p1"], details)
            self.assertEqual(store.read_pointer(pointer).value, key)
            snapshot = from_json_bytes(store.get(key))
            self.assertEqual(set(snapshot["manifests"]), {"p0", "p1"})
            self.assertEqual(set(snapshot["routing_entries"]), {"p0", "p1"})
            with self.assertRaisesRegex(RuntimeError, "publication conflict"):
                _publish_collection_snapshot(partitions, ["p0", "p1"], details)
            self.assertEqual(store.read_pointer(pointer).value, key)
            for partition_id in ("p0", "p1"):
                self.assertEqual(partitions.read_current_manifest(partition_id)[0].active_generation, "g0")
                candidate = partitions._read_manifest(snapshot["manifests"][partition_id])
                self.assertEqual(candidate.active_generation, "g1")
                self.assertIn("g0", candidate.retained_generations)

    def test_global_publication_rejects_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalObjectStore(tmpdir)
            partitions = PartitionStore(store, FlatIndex(), "test")
            partitions.create_partition("p0", "g0", [VectorRecord(id="a", vector=[1.0])])
            migration = MigrationExecutor(partitions, RoutingStore(store, "test"))
            prepared = migration.migrate_partition("p0", "g1", IdentityTransformer(), prepare_only=True)
            migration.migrate_partition("p0", "g2", IdentityTransformer())
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                _publish_collection_snapshot(partitions, ["p0"], [{
                    "partition_id": "p0", "source_version": prepared.source_version,
                    "prepared_manifest_key": prepared.prepared_manifest_key,
                }])
            self.assertIsNone(store.read_pointer("datasets/test/collection/generation.current").value)

    def test_thread_budget_and_largest_first_helpers(self) -> None:
        self.assertEqual(_faiss_threads_per_worker(16, 1), 16)
        self.assertEqual(_faiss_threads_per_worker(16, 2), 8)
        self.assertEqual(_faiss_threads_per_worker(16, 4), 4)
        self.assertEqual(_faiss_threads_per_worker(16, 8), 2)
        self.assertEqual(
            _largest_partitions_first(
                ["p0002", "p0000", "p0001"],
                {"p0000": 7, "p0001": 7, "p0002": 10},
            ),
            ["p0002", "p0000", "p0001"],
        )

    def test_tiny_evolution_benchmark_reports_progress_and_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "sift"
            dataset.mkdir()
            left = np.asarray(
                [[float(index) / 100.0, 0.0] for index in range(20)],
                dtype=np.float32,
            )
            right = np.asarray(
                [[10.0 + float(index) / 100.0, 0.0] for index in range(20)],
                dtype=np.float32,
            )
            base = np.concatenate([left, right])
            _write_fvecs(dataset / "sift_base.fvecs", base)

            config = EvolutionBenchmarkConfig(
                dataset_dir=dataset,
                database_limit=None,
                partition_count=2,
                sample_fraction=0.5,
                ingest_batch_size=10,
                ingest_clusters=2,
                ingest_max_iter=10,
                ingest_n_init=1,
                seed=3,
                faiss_niter=5,
                source_index_backend="flat",
                hnsw_m=8,
                hnsw_ef_construction=40,
                ivf_nlist=2,
                ivf_niter=5,
                keep_data=True,
                migration_workers=(1, 2),
                faiss_thread_budget=2,
                resource_sample_interval=0.005,
                work_parent=root,
            )
            with self.assertRaisesRegex(ValueError, "migration_workers"):
                replace(
                    config,
                    migration_workers=(2, 1),
                ).validate()

            with self.assertRaisesRegex(ValueError, "milestones"):
                replace(
                    config,
                    migration_milestones=(0.5, 1.0),
                ).validate()

            with self.assertRaisesRegex(ValueError, "positive"):
                replace(config, faiss_thread_budget=0).validate()

            with self.assertRaisesRegex(ValueError, "exceed"):
                replace(config, faiss_thread_budget=1).validate()

            result = run_evolution_benchmark(config)

            self.assertEqual(
                result["source_setup"]["layout"],
                {
                    "partition_count": 2,
                    "base_records": 40,
                    "delta_segments": 0,
                    "delta_records": 0,
                    "partition_vector_counts": {
                        "p0000": 20,
                        "p0001": 20,
                    },
                    "partition_vector_stats": {
                        "min": 20,
                        "max": 20,
                        "mean": 20.0,
                        "stddev": 0.0,
                        "p50": 20.0,
                        "p95": 20.0,
                        "imbalance_ratio": 1.0,
                    },
                },
            )
            self.assertNotIn("query_vectors", result["dataset"])
            self.assertNotIn("ground_truth_seconds", result["dataset"])
            self.assertNotIn("ground_truth_source", result["dataset"])
            self.assertEqual(
                set(result["strategies"]),
                {
                    "full-ivf",
                    "full-hnsw",
                    "partition-migration-w1",
                    "partition-migration-w2",
                    "partition-global-publish-w1",
                    "partition-global-publish-w2",
                },
            )
            for name in ("full-ivf", "full-hnsw"):
                strategy = result["strategies"][name]
                published = strategy["published_milestones"]
                self.assertEqual(strategy["record_count"], 40)
                self.assertEqual(
                    published["time_to_25_percent_seconds"],
                    strategy["total_seconds"],
                )
                output_root = Path(result["work_directory"]) / name / "g1"
                vectors = decode_float32_vectors(
                    (output_root / "vectors.npy").read_bytes()
                )
                ids = from_json_bytes((output_root / "ids.json").read_bytes())
                self.assertEqual(len(vectors), len(ids))
                self.assertEqual(vectors.shape, (40, 2))
                self.assertEqual(vectors.dtype, np.dtype("<f4"))
                self.assertFalse((output_root / "vectors.json").exists())
                self.assertFalse((output_root / "vectors.f32").exists())
                self.assertGreater(strategy["records_per_second"], 0.0)
                self.assertGreater(strategy["resources"]["peak_rss_bytes"], 0)
                self.assertNotIn("query_metrics", strategy)
                self.assertEqual(strategy["faiss_thread_budget"], 2)
                self.assertEqual(strategy["faiss_threads_per_worker"], 2)
                self.assertEqual(strategy["faiss_effective_thread_budget"], 2)

            expected_phases = {
                "read",
                "transform",
                "index_build",
                "serialize",
                "write",
                "publish",
            }
            for worker_count in (1, 2):
                strategy = result["strategies"][f"partition-global-publish-w{worker_count}"]
                self.assertEqual(strategy["publication_mode"], "global")
                for metric in ("time_to_first_partition_seconds", "time_to_25_percent_seconds",
                               "time_to_50_percent_seconds", "time_to_100_percent_seconds"):
                    self.assertEqual(strategy[metric], strategy["total_seconds"])
                self.assertEqual(strategy["preparation_milestones"]["100"]["g0_partition_count"], 2)
                self.assertLessEqual(strategy["preparation_seconds"], strategy["total_seconds"])
                self.assertEqual(set(strategy["phase_worker_seconds"]), expected_phases)
                output = Path(result["work_directory"]) / strategy["strategy"]
                snapshot = from_json_bytes((output / strategy["collection_snapshot_key"]).read_bytes())
                self.assertEqual(len(snapshot["manifests"]), 2)
            for worker_count in (1, 2):
                migration = result["strategies"][
                    f"partition-migration-w{worker_count}"
                ]
                self.assertEqual(migration["migration_workers"], worker_count)
                expected_threads = 2 // worker_count
                self.assertEqual(migration["faiss_thread_budget"], 2)
                self.assertEqual(
                    migration["faiss_threads_per_worker"],
                    expected_threads,
                )
                self.assertEqual(
                    migration["faiss_effective_thread_budget"],
                    2,
                )
                for metric in (
                    "phase_worker_seconds",
                    "phase_mean_seconds",
                    "phase_p95_seconds",
                    "phase_max_seconds",
                ):
                    self.assertEqual(set(migration[metric]), expected_phases)
                    self.assertTrue(all(value >= 0.0 for value in migration[metric].values()))
                for detail in migration["partitions"]:
                    self.assertEqual(detail["faiss_threads"], expected_threads)
                    self.assertEqual(set(detail["phase_seconds"]), expected_phases)
                    self.assertLessEqual(
                        sum(detail["phase_seconds"].values()), detail["instrumented_total_seconds"]
                    )
                self.assertEqual(set(migration["milestones"]), {"25", "50", "75", "100"})
                self.assertGreater(migration["time_to_first_partition_seconds"], 0.0)
                counts = {
                    detail["partition_id"]: detail["record_count"]
                    for detail in migration["partitions"]
                }
                self.assertEqual(
                    migration["scheduled_partition_ids"],
                    _largest_partitions_first(list(counts), counts),
                )
                self.assertLessEqual(
                    migration["time_to_25_percent_seconds"],
                    migration["time_to_50_percent_seconds"],
                )
                self.assertLessEqual(
                    migration["time_to_50_percent_seconds"],
                    migration["time_to_100_percent_seconds"],
                )
                self.assertLessEqual(
                    migration["time_to_100_percent_seconds"],
                    migration["total_seconds"],
                )
                final = migration["milestones"]["100"]
                self.assertEqual(final["g0_partition_count"], 0)
                self.assertEqual(final["g1_partition_count"], 2)
                self.assertNotIn("query_metrics", final)
                self.assertNotIn("query_checkpoints", migration)
                self.assertGreaterEqual(
                    migration["resources"]["peak_temporary_storage_bytes"],
                    0,
                )


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype="<f4")
    framed = np.empty((len(vectors), vectors.shape[1] + 1), dtype="<i4")
    framed[:, 0] = vectors.shape[1]
    framed[:, 1:] = vectors.view("<i4")
    framed.tofile(path)


if __name__ == "__main__":
    unittest.main()
