"""Tests for the SIFT benchmark runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.sift_benchmark import (
    BenchmarkConfig,
    _format_elapsed,
    _parser,
    read_fvecs,
    recall_at_k,
    run_benchmark,
)


class SiftBenchmarkTest(unittest.TestCase):
    """Covers vector readers, recall, and all benchmark layouts."""

    def test_hnsw_is_the_default_index_backend(self) -> None:
        self.assertEqual(BenchmarkConfig().index_backend, "faiss-hnsw")
        self.assertEqual(BenchmarkConfig().dataset_prefix, "sift")
        self.assertEqual(_parser().parse_args([]).index_backend, "faiss-hnsw")
        self.assertEqual(BenchmarkConfig().query_partition_workers, 8)
        self.assertEqual(BenchmarkConfig().query_object_workers, 16)
        self.assertEqual(_parser().parse_args([]).query_partition_workers, 8)
        self.assertEqual(_parser().parse_args([]).query_object_workers, 16)

    def test_elapsed_time_format(self) -> None:
        self.assertEqual(_format_elapsed(3661.2349), "01:01:01.234")
        self.assertEqual(_format_elapsed(-1.0), "00:00:00.000")

    def test_read_fvecs_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.fvecs"
            expected = np.asarray(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            )
            _write_fvecs(path, expected)

            actual = read_fvecs(path)

            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(
                recall_at_k(
                    [[1, 2], [4, 9]],
                    np.asarray([[1, 3], [4, 5]], dtype=np.int64),
                    2,
                ),
                0.5,
            )

    def test_tiny_benchmark_runs_all_scenarios(self) -> None:
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
            queries = np.asarray(
                [[0.05, 0.0], [0.15, 0.0], [10.05, 0.0], [10.15, 0.0]],
                dtype=np.float32,
            )
            _write_fvecs(dataset / "sift_base.fvecs", base)
            _write_fvecs(dataset / "sift_query.fvecs", queries)

            config = BenchmarkConfig(
                dataset_dir=dataset,
                database_limit=None,
                query_limit=4,
                partition_count=2,
                sample_fraction=0.5,
                ingest_batch_size=10,
                ingest_clusters=2,
                ingest_max_iter=10,
                ingest_n_init=1,
                top_k=2,
                routing_top_ks=(1, None),
                warmup_queries=0,
                compact=True,
                seed=3,
                faiss_niter=5,
                work_parent=root,
            )
            with self.assertRaisesRegex(
                ValueError,
                "batch-local bootstrap",
            ):
                replace(config, ingest_batch_size=1).validate()
            progress: list[str] = []
            result = run_benchmark(config, progress=progress.append)

            self.assertEqual(result["dataset"]["database_vectors"], 40)
            expected_progress = [
                "dataset loaded",
                "ground truth computed",
                "current layout started",
                "current sample layout completed",
                "current ingest completed",
                "current uncompacted query started: fanout=1",
                "current compacted query started: fanout=all",
                "current layout completed",
                "batch-local semantic baseline started",
                "batch-local semantic ingest completed",
                "global-clustering baseline started",
                "global-clustering query started: fanout=1",
                "global-clustering baseline completed",
                "block baseline started",
                "block query started: fanout=all",
                "block baseline completed",
                "benchmark completed",
            ]
            progress_positions = [
                next(
                    index
                    for index, message in enumerate(progress)
                    if message.startswith(prefix)
                )
                for prefix in expected_progress
            ]
            self.assertEqual(
                result["dataset"]["ground_truth_source"],
                "faiss.IndexFlatL2",
            )
            scenarios = result["scenarios"]
            self.assertEqual(
                set(scenarios),
                {
                    "current",
                    "batch_local_semantic",
                    "global_clustering",
                    "block",
                },
            )
            self.assertGreater(
                scenarios["current"]["uncompacted"]["layout"]["delta_segments"],
                0,
            )
            self.assertEqual(
                scenarios["batch_local_semantic"]["bootstrap_records"],
                config.ingest_batch_size,
            )
            self.assertGreater(
                scenarios["batch_local_semantic"]["uncompacted"]["layout"][
                    "delta_segments"
                ],
                0,
            )
            batch_local_layout = scenarios["batch_local_semantic"][
                "uncompacted"
            ]["layout"]
            self.assertEqual(
                batch_local_layout["base_records"]
                + batch_local_layout["delta_records"],
                len(base),
            )
            remaining_batch_count = (
                len(base) - config.ingest_batch_size
                + config.ingest_batch_size - 1
            ) // config.ingest_batch_size
            self.assertLessEqual(
                batch_local_layout["delta_segments"],
                remaining_batch_count * config.partition_count,
            )
            self.assertEqual(
                scenarios["current"]["compacted"]["layout"]["delta_segments"],
                0,
            )
            self.assertEqual(
                scenarios["batch_local_semantic"]["compacted"]["layout"][
                    "delta_segments"
                ],
                0,
            )
            query_states = [
                scenarios["current"]["uncompacted"],
                scenarios["current"]["compacted"],
                scenarios["batch_local_semantic"]["uncompacted"],
                scenarios["batch_local_semantic"]["compacted"],
                scenarios["global_clustering"],
                scenarios["block"],
            ]
            for query_state in query_states:
                layout = query_state["layout"]
                counts = layout["partition_vector_counts"]
                sizes = np.asarray(list(counts.values()), dtype=np.float64)
                stats = layout["partition_vector_stats"]
                self.assertEqual(len(counts), layout["partition_count"])
                self.assertEqual(sum(counts.values()), len(base))
                self.assertEqual(stats["min"], int(np.min(sizes)))
                self.assertEqual(stats["max"], int(np.max(sizes)))
                self.assertAlmostEqual(stats["mean"], float(np.mean(sizes)))
                self.assertAlmostEqual(stats["stddev"], float(np.std(sizes)))
                self.assertAlmostEqual(
                    stats["p50"],
                    float(np.percentile(sizes, 50)),
                )
                self.assertAlmostEqual(
                    stats["p95"],
                    float(np.percentile(sizes, 95)),
                )
                self.assertAlmostEqual(
                    stats["imbalance_ratio"],
                    float(np.max(sizes) / np.mean(sizes)),
                )
                self.assertGreaterEqual(
                    query_state["routing_checkpoint_seconds"],
                    0.0,
                )
                self.assertGreater(query_state["routing_checkpoint_epoch"], 0)
                query_metrics = query_state["queries"]
                self.assertEqual(query_metrics["all"]["recall_at_k"], 1.0)
                self.assertGreater(query_metrics["all"]["qps"], 0.0)
                metrics = query_metrics["all"]
                expected_phases = {
                    "routing_read",
                    "routing_compute",
                    "manifest_read",
                    "tombstone_read",
                    "ids_read",
                    "index_read",
                    "index_deserialize",
                    "ann_search",
                    "visibility_merge",
                    "global_merge",
                    "other",
                }
                self.assertEqual(set(metrics["mean_breakdown_ms"]), expected_phases)
                self.assertEqual(set(metrics["p95_breakdown_ms"]), expected_phases)
                self.assertEqual(metrics["phase_accounting"], "worker-seconds")
                self.assertGreater(sum(metrics["mean_breakdown_ms"].values()), 0.0)
                self.assertGreater(metrics["mean_selected_partitions"], 0.0)
                self.assertGreater(metrics["mean_loaded_segments"], 0.0)
            self.assertEqual(
                scenarios["current"]["uncompacted"]["layout"][
                    "partition_vector_counts"
                ],
                scenarios["current"]["compacted"]["layout"][
                    "partition_vector_counts"
                ],
            )
            self.assertEqual(
                scenarios["batch_local_semantic"]["uncompacted"]["layout"][
                    "partition_vector_counts"
                ],
                scenarios["batch_local_semantic"]["compacted"]["layout"][
                    "partition_vector_counts"
                ],
            )
            self.assertGreater(
                scenarios["current"]["compacted"]["routing_checkpoint_epoch"],
                scenarios["current"]["uncompacted"]["routing_checkpoint_epoch"],
            )
            self.assertGreater(
                scenarios["batch_local_semantic"]["compacted"][
                    "routing_checkpoint_epoch"
                ],
                scenarios["batch_local_semantic"]["uncompacted"][
                    "routing_checkpoint_epoch"
                ],
            )
            hnsw_result = run_benchmark(replace(
                config,
                compact=False,
                hnsw_ef_construction=40,
                hnsw_ef_search=20,
                hnsw_m=8,
                index_backend="faiss-hnsw",
                routing_top_ks=(None,),
            ))
            self.assertEqual(
                hnsw_result["config"]["index_backend"],
                "faiss-hnsw",
            )
            for scenario in hnsw_result["scenarios"].values():
                query_state = (
                    scenario["uncompacted"]
                    if "uncompacted" in scenario
                    else scenario
                )
                self.assertGreaterEqual(
                    query_state["routing_checkpoint_seconds"],
                    0.0,
                )
                self.assertGreater(query_state["routing_checkpoint_epoch"], 0)
                self.assertEqual(
                    query_state["queries"]["all"]["recall_at_k"],
                    1.0,
                )


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype="<f4")
    framed = np.empty(
        (len(vectors), vectors.shape[1] + 1),
        dtype="<i4",
    )
    framed[:, 0] = vectors.shape[1]
    framed[:, 1:] = vectors.view("<i4")
    framed.tofile(path)


if __name__ == "__main__":
    unittest.main()
