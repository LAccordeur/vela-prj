"""Tests for compacted-layout single- and multi-centroid routing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.sift_routing_benchmark import (
    RoutingBenchmarkConfig,
    run_routing_benchmark,
    select_partitions,
)


class SiftRoutingBenchmarkTest(unittest.TestCase):
    """Covers nearest-centroid routing and the end-to-end benchmark."""

    def test_multi_centroid_can_change_the_best_partition(self) -> None:
        query = [9.0]
        single = {
            "p0000": np.asarray([[5.0]], dtype=np.float32),
            "p0001": np.asarray([[6.0]], dtype=np.float32),
        }
        multi = {
            "p0000": np.asarray([[0.0], [10.0]], dtype=np.float32),
            "p0001": np.asarray([[4.0], [6.0]], dtype=np.float32),
        }

        self.assertEqual(
            select_partitions(query, single, 1),
            ["p0001"],
        )
        self.assertEqual(
            select_partitions(query, multi, 1),
            ["p0000"],
        )
        self.assertEqual(
            select_partitions(query, multi, None),
            ["p0000", "p0001"],
        )

    def test_tiny_routing_benchmark_compares_centroid_counts(self) -> None:
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

            config = RoutingBenchmarkConfig(
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
                centroid_counts=(1, 2),
                warmup_queries=0,
                seed=3,
                faiss_niter=5,
                centroid_niter=5,
                index_backend="flat",
                work_parent=root,
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                replace(config, centroid_counts=(1, 1)).validate()

            result = run_routing_benchmark(config)

            self.assertEqual(
                result["dataset"]["ground_truth_source"],
                "faiss.IndexFlatL2",
            )
            self.assertEqual(
                result["layout"]["objects"],
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
            self.assertEqual(set(result["strategies"]), {"1", "2"})
            self.assertEqual(
                result["strategies"]["1"]["total_centroids"],
                2,
            )
            self.assertEqual(
                result["strategies"]["2"]["total_centroids"],
                4,
            )
            for strategy in result["strategies"].values():
                all_metrics = strategy["measurements"]["all"]
                self.assertEqual(all_metrics["recall_at_k"], 1.0)
                self.assertEqual(
                    all_metrics["mean_selected_partitions"],
                    2.0,
                )
                self.assertGreater(all_metrics["qps"], 0.0)
                self.assertGreaterEqual(
                    all_metrics["total_mean_ms"],
                    all_metrics["routing_mean_ms"],
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
