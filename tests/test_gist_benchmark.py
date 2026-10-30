"""Tests for the GIST construction benchmark entry point."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.gist_benchmark import _parser
from benchmarks.sift_benchmark import BenchmarkConfig, run_benchmark


class GistBenchmarkTest(unittest.TestCase):
    """Covers GIST defaults, file discovery, and official ground truth."""

    def test_cli_defaults_target_gist(self) -> None:
        args = _parser().parse_args([])

        self.assertEqual(
            args.dataset_dir,
            Path("/home/yangguo/Dataset/gist"),
        )
        self.assertEqual(args.database_limit, 10_000)
        self.assertEqual(args.index_backend, "faiss-hnsw")

    def test_tiny_gist_benchmark_uses_official_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "gist"
            dataset.mkdir()
            rng = np.random.default_rng(7)
            base = rng.random((80, 4), dtype=np.float32)
            queries = base[[0, 10, 20, 30]].copy()
            ground_truth = np.asarray(
                [[0, 1], [10, 11], [20, 21], [30, 31]],
                dtype=np.int32,
            )
            _write_fvecs(dataset / "gist_base.fvecs", base)
            _write_fvecs(dataset / "gist_query.fvecs", queries)
            _write_ivecs(
                dataset / "gist_groundtruth.ivecs",
                ground_truth,
            )

            config = BenchmarkConfig(
                dataset_dir=dataset,
                dataset_prefix="gist",
                database_limit=None,
                query_limit=4,
                partition_count=2,
                sample_fraction=0.5,
                ingest_batch_size=10,
                ingest_clusters=2,
                ingest_max_iter=5,
                ingest_n_init=1,
                top_k=2,
                routing_top_ks=(1, None),
                warmup_queries=0,
                compact=False,
                seed=7,
                faiss_niter=3,
                faiss_nredo=1,
                index_backend="flat",
                query_partition_workers=1,
                query_object_workers=1,
                work_parent=root,
            )
            result = run_benchmark(config)

            self.assertEqual(result["dataset"]["name"], "gist")
            self.assertEqual(result["dataset"]["dimension"], 4)
            self.assertEqual(
                result["dataset"]["ground_truth_source"],
                "gist_groundtruth.ivecs",
            )
            self.assertEqual(
                set(result["scenarios"]),
                {
                    "current",
                    "batch_local_semantic",
                    "global_clustering",
                    "block",
                },
            )


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    rows = np.asarray(vectors, dtype="<f4")
    framed = np.empty((len(rows), rows.shape[1] + 1), dtype="<i4")
    framed[:, 0] = rows.shape[1]
    framed[:, 1:] = rows.view("<i4")
    framed.tofile(path)


def _write_ivecs(path: Path, vectors: np.ndarray) -> None:
    rows = np.asarray(vectors, dtype="<i4")
    framed = np.empty((len(rows), rows.shape[1] + 1), dtype="<i4")
    framed[:, 0] = rows.shape[1]
    framed[:, 1:] = rows
    framed.tofile(path)


if __name__ == "__main__":
    unittest.main()
