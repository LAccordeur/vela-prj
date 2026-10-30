"""SIFT benchmark for incremental Vela and offline partitioning baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable

import faiss
import numpy as np

from benchmarks.storage import (
    SUPPORTED_STORAGE_BACKENDS,
    clear_benchmark_store,
    create_benchmark_store,
    describe_benchmark_storage,
    resolve_storage_namespace,
)
from vela.execution.baseline import BaselineExecutor
from vela.execution.bootstrap import BootstrapExecutor
from vela.execution.compaction import CompactionExecutor
from vela.execution.ingest import IngestExecutor
from vela.execution.query import QueryExecutor, QueryProfile
from vela.index.base import ANNIndex
from vela.index.faiss_hnsw import FaissHNSWIndex
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.partition.kmeans_policy import KMeansPartitionPolicy
from vela.partition.policy import PartitionTarget
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for one reproducible vector benchmark run."""

    dataset_dir: Path = Path("/home/yangguo/Dataset/sift")
    database_limit: int | None = 10_000
    query_limit: int = 100
    partition_count: int = 16
    sample_fraction: float = 0.1
    ingest_batch_size: int = 1_000
    ingest_clusters: int = 16
    ingest_max_iter: int = 100
    ingest_n_init: int = 1
    top_k: int = 10
    routing_top_ks: tuple[int | None, ...] = (1, 4, 8, None)
    warmup_queries: int = 3
    compact: bool = True
    seed: int = 0
    faiss_niter: int = 25
    faiss_nredo: int = 1
    index_backend: str = "faiss-hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    query_partition_workers: int = 8
    query_object_workers: int = 16
    storage_backend: str = "local"
    storage_namespace: str | None = None
    work_parent: Path | None = None
    keep_data: bool = False
    dataset_prefix: str = "sift"

    def validate(self) -> None:
        """Reject configurations that cannot produce a meaningful comparison."""

        normalized_prefix = (
            self.dataset_prefix.replace("-", "").replace("_", "")
        )
        if not normalized_prefix or not normalized_prefix.isalnum():
            raise ValueError(
                "dataset_prefix must contain only letters, numbers, '-' or '_'"
            )
        if self.database_limit is not None and self.database_limit < 1:
            raise ValueError("database_limit must be positive or None")
        if self.query_limit < 1:
            raise ValueError("query_limit must be positive")
        if self.partition_count < 1:
            raise ValueError("partition_count must be positive")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.ingest_batch_size < 1 or self.ingest_clusters < 1:
            raise ValueError("ingest batch size and cluster count must be positive")
        if self.ingest_batch_size < self.partition_count:
            raise ValueError(
                "ingest_batch_size must be at least partition_count "
                "for batch-local bootstrap"
            )
        if self.ingest_max_iter < 1 or self.ingest_n_init < 1:
            raise ValueError("ingest K-Means settings must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.warmup_queries < 0:
            raise ValueError("warmup_queries must not be negative")
        if self.faiss_niter < 1 or self.faiss_nredo < 1:
            raise ValueError("FAISS K-Means settings must be positive")
        if self.index_backend not in {"flat", "faiss-hnsw"}:
            raise ValueError("index_backend must be flat or faiss-hnsw")
        if self.storage_backend not in SUPPORTED_STORAGE_BACKENDS:
            raise ValueError(
                f"storage_backend must be one of {SUPPORTED_STORAGE_BACKENDS}"
            )
        if (
            self.hnsw_m < 1
            or self.hnsw_ef_construction < 1
            or self.hnsw_ef_search < 1
        ):
            raise ValueError("HNSW settings must be positive")
        if self.query_partition_workers < 1 or self.query_object_workers < 1:
            raise ValueError("query worker settings must be positive")
        concrete_fanouts = [value for value in self.routing_top_ks if value is not None]
        if any(value < 1 for value in concrete_fanouts):
            raise ValueError("routing_top_k values must be positive or all")


@dataclass
class CurrentLayoutBuild:
    """One sampled incremental layout ready for query or compaction."""

    index: ANNIndex
    partition_store: PartitionStore
    routing_store: RoutingStore
    partition_ids: list[str]
    sample_seconds: float
    ingest_seconds: float
    sample_count: int
    ingested_count: int

    @property
    def load_seconds(self) -> float:
        """Return bootstrap plus ingest wall time."""

        return self.sample_seconds + self.ingest_seconds


def read_fvecs(path: Path, limit: int | None = None) -> np.ndarray:
    """Read a standard little-endian fvecs file into a float32 matrix."""

    dimension, rows = _framed_vector_shape(path)
    count = rows if limit is None else min(limit, rows)
    raw = np.memmap(path, dtype="<i4", mode="r").reshape(rows, dimension + 1)
    if not np.all(raw[:count, 0] == dimension):
        raise ValueError(f"inconsistent vector dimensions in {path}")
    return np.array(raw[:count, 1:].view("<f4"), dtype=np.float32, copy=True)


def read_ivecs(path: Path, limit: int | None = None) -> np.ndarray:
    """Read a standard little-endian ivecs file into an int64 matrix."""

    dimension, rows = _framed_vector_shape(path)
    count = rows if limit is None else min(limit, rows)
    raw = np.memmap(path, dtype="<i4", mode="r").reshape(rows, dimension + 1)
    if not np.all(raw[:count, 0] == dimension):
        raise ValueError(f"inconsistent vector dimensions in {path}")
    return np.array(raw[:count, 1:], dtype=np.int64, copy=True)


def recall_at_k(
    predicted: list[list[int]],
    ground_truth: np.ndarray,
    top_k: int,
) -> float:
    """Return mean set recall over the first top_k neighbors of each query."""

    if len(predicted) != len(ground_truth):
        raise ValueError("predicted and ground-truth query counts must match")
    if not predicted:
        return 0.0
    total = 0.0
    for actual, expected in zip(predicted, ground_truth, strict=True):
        expected_ids = {int(value) for value in expected[:top_k]}
        actual_ids = set(actual[:top_k])
        total += len(actual_ids & expected_ids) / top_k
    return total / len(predicted)


def run_benchmark(
    config: BenchmarkConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build all four layouts and measure load, query, recall, and compaction."""

    config = replace(
        config,
        storage_namespace=resolve_storage_namespace(
            config.storage_backend,
            config.storage_namespace,
            f"{config.dataset_prefix}-construction",
        ),
    )
    config.validate()
    load_started = time.perf_counter()
    base_path = config.dataset_dir / f"{config.dataset_prefix}_base.fvecs"
    query_path = config.dataset_dir / f"{config.dataset_prefix}_query.fvecs"
    base_vectors = read_fvecs(base_path, config.database_limit)
    query_vectors = read_fvecs(query_path, config.query_limit)
    dataset_load_seconds = time.perf_counter() - load_started
    _report(progress, f"dataset loaded ({dataset_load_seconds:.3f}s)")

    if config.partition_count > len(base_vectors):
        raise ValueError("partition_count must not exceed loaded database size")
    if config.top_k > len(base_vectors):
        raise ValueError("top_k must not exceed loaded database size")

    ground_truth_started = time.perf_counter()
    ground_truth, ground_truth_source = compute_ground_truth(
        config,
        base_vectors,
        query_vectors,
    )
    ground_truth_seconds = time.perf_counter() - ground_truth_started
    _report(progress, f"ground truth computed ({ground_truth_seconds:.3f}s)")
    query_lists = [vector.tolist() for vector in query_vectors]

    work_parent = str(config.work_parent) if config.work_parent is not None else None
    root = Path(tempfile.mkdtemp(
        prefix=f"vela-{config.dataset_prefix}-",
        dir=work_parent,
    ))
    result: dict[str, Any] = {
        "config": _json_config(config),
        "dataset": {
            "name": config.dataset_prefix,
            "database_vectors": len(base_vectors),
            "query_vectors": len(query_vectors),
            "dimension": int(base_vectors.shape[1]),
            "read_seconds": dataset_load_seconds,
            "ground_truth_seconds": ground_truth_seconds,
            "ground_truth_source": ground_truth_source,
        },
        "storage": describe_benchmark_storage(
            config.storage_backend,
            config.storage_namespace,
        ),
        "work_directory": str(root) if config.keep_data else None,
        "scenarios": {},
    }
    try:
        _report(progress, "current layout started")
        result["scenarios"]["current"] = _run_current(
            root / "current",
            base_vectors,
            query_lists,
            ground_truth,
            config,
            progress,
        )
        _report(progress, "current layout completed")
        gc.collect()
        _report(progress, "batch-local semantic baseline started")
        result["scenarios"]["batch_local_semantic"] = (
            _run_batch_local_semantic(
                root / "batch-local-semantic",
                base_vectors,
                query_lists,
                ground_truth,
                config,
                progress,
            )
        )
        _report(progress, "batch-local semantic baseline completed")
        gc.collect()
        _report(progress, "global-clustering baseline started")
        result["scenarios"]["global_clustering"] = _run_global_clustering(
            root / "global-clustering",
            base_vectors,
            query_lists,
            ground_truth,
            config,
            progress,
        )
        _report(progress, "global-clustering baseline completed")
        gc.collect()
        _report(progress, "block baseline started")
        result["scenarios"]["block"] = _run_block(
            root / "block",
            base_vectors,
            query_lists,
            ground_truth,
            config,
            progress,
        )
        _report(progress, "block baseline completed")
    finally:
        if not config.keep_data:
            if config.storage_backend == "s3":
                clear_benchmark_store(
                    create_benchmark_store(
                        config.storage_backend,
                        root,
                        config.storage_namespace,
                        "",
                    ),
                    root,
                )
            shutil.rmtree(root, ignore_errors=True)
    _report(progress, "benchmark completed")
    return result


def _run_current(
    root: Path,
    vectors: np.ndarray,
    queries: list[list[float]],
    ground_truth: np.ndarray,
    config: BenchmarkConfig,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    layout = build_current_layout(
        root,
        vectors,
        config,
        progress=progress,
    )
    query = _new_query_executor(
        layout.partition_store,
        layout.routing_store,
        layout.index,
        config,
    )
    checkpoint_seconds, checkpoint_epoch = _publish_routing_checkpoint(
        layout.routing_store
    )
    current: dict[str, Any] = {
        "sample_build_seconds": layout.sample_seconds,
        "ingest_seconds": layout.ingest_seconds,
        "load_seconds": layout.load_seconds,
        "sample_records": layout.sample_count,
        "ingested_records": layout.ingested_count,
        "uncompacted": {
            "routing_checkpoint_seconds": checkpoint_seconds,
            "routing_checkpoint_epoch": checkpoint_epoch,
            "layout": layout_stats(layout.partition_store),
            "queries": _measure_queries(
                query,
                queries,
                ground_truth,
                config,
                progress=progress,
                query_label="current uncompacted",
            ),
        },
    }
    if config.compact:
        current["compaction_seconds"] = compact_current_layout(layout)
        _report(
            progress,
            "current compaction completed "
            f"({current['compaction_seconds']:.3f}s)",
        )
        checkpoint_seconds, checkpoint_epoch = _publish_routing_checkpoint(
            layout.routing_store
        )
        current["compacted"] = {
            "routing_checkpoint_seconds": checkpoint_seconds,
            "routing_checkpoint_epoch": checkpoint_epoch,
            "layout": layout_stats(layout.partition_store),
            "queries": _measure_queries(
                query,
                queries,
                ground_truth,
                config,
                progress=progress,
                query_label="current compacted",
            ),
        }
    query.close()
    return current


def build_current_layout(
    root: Path,
    vectors: np.ndarray,
    config: BenchmarkConfig,
    *,
    dataset_id: str = "current",
    generation: str = "g0",
    progress: ProgressCallback | None = None,
) -> CurrentLayoutBuild:
    """Build the sampled base plus batch-ingested current Vela layout."""

    index, partition_store, routing_store = _system(
        root,
        dataset_id,
        config,
    )
    rng = np.random.default_rng(config.seed)
    sample_count = max(
        config.partition_count,
        int(round(len(vectors) * config.sample_fraction)),
    )
    sample_count = min(sample_count, len(vectors))
    sample_indices = np.sort(
        rng.choice(len(vectors), size=sample_count, replace=False)
    )
    remaining_mask = np.ones(len(vectors), dtype=bool)
    remaining_mask[sample_indices] = False
    remaining_indices = np.flatnonzero(remaining_mask)

    sample_started = time.perf_counter()
    sample_records = _records(vectors, sample_indices)
    build_result = BaselineExecutor(
        partition_store,
        routing_store,
    ).build_clustering(
        sample_records,
        config.partition_count,
        generation,
        seed=config.seed,
        niter=config.faiss_niter,
        nredo=config.faiss_nredo,
    )
    sample_seconds = time.perf_counter() - sample_started
    _report(progress, f"current sample layout completed ({sample_seconds:.3f}s)")
    del sample_records

    ingest = IngestExecutor(
        partition_store,
        routing_store,
        KMeansPartitionPolicy(
            n_clusters=config.ingest_clusters,
            random_state=config.seed,
            max_iter=config.ingest_max_iter,
            n_init=config.ingest_n_init,
        ),
    )
    ingest_started = time.perf_counter()
    for batch_indices in _batches(remaining_indices, config.ingest_batch_size):
        ingest_result = ingest.ingest(
            _records(vectors, batch_indices),
            candidate_partitions=build_result.partition_ids,
        )
        if ingest_result.conflicts:
            raise RuntimeError(
                f"ingest conflicts in benchmark: {ingest_result.conflicts}"
            )
    ingest_seconds = time.perf_counter() - ingest_started
    _report(progress, f"current ingest completed ({ingest_seconds:.3f}s)")
    return CurrentLayoutBuild(
        index=index,
        partition_store=partition_store,
        routing_store=routing_store,
        partition_ids=build_result.partition_ids,
        sample_seconds=sample_seconds,
        ingest_seconds=ingest_seconds,
        sample_count=sample_count,
        ingested_count=len(remaining_indices),
    )


def compact_current_layout(layout: CurrentLayoutBuild) -> float:
    """Compact every partition in a built current layout and return wall time."""

    compact_started = time.perf_counter()
    compactor = CompactionExecutor(
        layout.partition_store,
        layout.routing_store,
    )
    for partition_id in layout.partition_ids:
        compact_result = compactor.compact_partition(partition_id)
        if compact_result.conflict:
            raise RuntimeError(
                f"compaction conflict in benchmark: {partition_id}"
            )
    return time.perf_counter() - compact_started
def _run_batch_local_semantic(
    root: Path,
    vectors: np.ndarray,
    queries: list[list[float]],
    ground_truth: np.ndarray,
    config: BenchmarkConfig,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    """Build partitions from independent batch-local K-Means models."""

    index, partition_store, routing_store = _system(
        root,
        "batch-local-semantic",
        config,
    )
    partition_ids = [
        f"p{partition_index:04d}"
        for partition_index in range(config.partition_count)
    ]
    counts = {partition_id: 0 for partition_id in partition_ids}
    sums = {
        partition_id: np.zeros(vectors.shape[1], dtype=np.float64)
        for partition_id in partition_ids
    }
    all_indices = np.arange(len(vectors), dtype=np.int64)
    first_size = min(config.ingest_batch_size, len(vectors))
    first_records = _records(vectors, all_indices[:first_size])
    first_policy = _kmeans_policy(config, config.partition_count)

    bootstrap_started = time.perf_counter()
    first_assignments = first_policy.assign_batch(
        first_records,
        [
            PartitionTarget(partition_id=partition_id)
            for partition_id in partition_ids
        ],
    )
    first_groups = _group_assigned_records(first_records, first_assignments)
    bootstrap = BootstrapExecutor(partition_store, routing_store)
    for partition_id in partition_ids:
        bootstrap.create_partition(
            partition_id,
            "g0",
            first_groups.get(partition_id, []),
        )
    _update_running_centroids(
        counts,
        sums,
        first_records,
        first_assignments,
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    _report(
        progress,
        "batch-local semantic bootstrap completed "
        f"({bootstrap_seconds:.3f}s)",
    )

    append_policy = _kmeans_policy(config, config.ingest_clusters)
    direct_ingest = IngestExecutor(
        partition_store,
        routing_store,
        HashPartitionPolicy(),
    )
    ingest_started = time.perf_counter()
    for batch_indices in _batches(
        all_indices[first_size:],
        config.ingest_batch_size,
    ):
        records = _records(vectors, batch_indices)
        assignments = append_policy.assign_batch(
            records,
            _running_targets(partition_ids, counts, sums),
        )
        grouped = _group_assigned_records(records, assignments)
        for partition_id, partition_records in grouped.items():
            ingest_result = direct_ingest.ingest(
                partition_records,
                candidate_partitions=[partition_id],
            )
            if ingest_result.conflicts:
                raise RuntimeError(
                    "batch-local ingest conflicts: "
                    f"{ingest_result.conflicts}"
                )
        _update_running_centroids(counts, sums, records, assignments)
    ingest_seconds = time.perf_counter() - ingest_started
    _report(
        progress,
        f"batch-local semantic ingest completed ({ingest_seconds:.3f}s)",
    )

    query = _new_query_executor(partition_store, routing_store, index, config)
    checkpoint_seconds, checkpoint_epoch = _publish_routing_checkpoint(
        routing_store
    )
    result: dict[str, Any] = {
        "bootstrap_seconds": bootstrap_seconds,
        "ingest_seconds": ingest_seconds,
        "load_seconds": bootstrap_seconds + ingest_seconds,
        "bootstrap_records": first_size,
        "ingested_records": len(vectors) - first_size,
        "uncompacted": {
            "routing_checkpoint_seconds": checkpoint_seconds,
            "routing_checkpoint_epoch": checkpoint_epoch,
            "layout": layout_stats(partition_store),
            "queries": _measure_queries(
                query,
                queries,
                ground_truth,
                config,
                progress=progress,
                query_label="batch-local semantic uncompacted",
            ),
        },
    }
    if config.compact:
        compact_started = time.perf_counter()
        compactor = CompactionExecutor(partition_store, routing_store)
        for partition_id in partition_ids:
            compact_result = compactor.compact_partition(partition_id)
            if compact_result.conflict:
                raise RuntimeError(
                    "batch-local compaction conflict: "
                    f"{partition_id}"
                )
        result["compaction_seconds"] = time.perf_counter() - compact_started
        _report(
            progress,
            "batch-local semantic compaction completed "
            f"({result['compaction_seconds']:.3f}s)",
        )
        checkpoint_seconds, checkpoint_epoch = _publish_routing_checkpoint(
            routing_store
        )
        result["compacted"] = {
            "routing_checkpoint_seconds": checkpoint_seconds,
            "routing_checkpoint_epoch": checkpoint_epoch,
            "layout": layout_stats(partition_store),
            "queries": _measure_queries(
                query,
                queries,
                ground_truth,
                config,
                progress=progress,
                query_label="batch-local semantic compacted",
            ),
        }
    query.close()
    return result


def _run_global_clustering(
    root: Path,
    vectors: np.ndarray,
    queries: list[list[float]],
    ground_truth: np.ndarray,
    config: BenchmarkConfig,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    index, partition_store, routing_store = _system(root, "global-clustering", config)
    started = time.perf_counter()
    records = _records(vectors, range(len(vectors)))
    BaselineExecutor(partition_store, routing_store).build_clustering(
        records,
        config.partition_count,
        "g0",
        seed=config.seed,
        niter=config.faiss_niter,
        nredo=config.faiss_nredo,
    )
    load_seconds = time.perf_counter() - started
    _report(progress, f"global-clustering layout built ({load_seconds:.3f}s)")
    del records
    checkpoint_seconds, checkpoint_epoch = _publish_routing_checkpoint(
        routing_store
    )
    return {
        "load_seconds": load_seconds,
        "routing_checkpoint_seconds": checkpoint_seconds,
        "routing_checkpoint_epoch": checkpoint_epoch,
        "layout": layout_stats(partition_store),
        "queries": _measure_queries(
            _new_query_executor(partition_store, routing_store, index, config),
            queries,
            ground_truth,
            config,
            progress=progress,
            query_label="global-clustering",
        ),
    }


def _run_block(
    root: Path,
    vectors: np.ndarray,
    queries: list[list[float]],
    ground_truth: np.ndarray,
    config: BenchmarkConfig,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    index, partition_store, routing_store = _system(root, "block", config)
    started = time.perf_counter()
    records = _records(vectors, range(len(vectors)))
    BaselineExecutor(partition_store, routing_store).build_blocks(
        records,
        config.partition_count,
        "g0",
    )
    load_seconds = time.perf_counter() - started
    _report(progress, f"block layout built ({load_seconds:.3f}s)")
    del records
    checkpoint_seconds, checkpoint_epoch = _publish_routing_checkpoint(
        routing_store
    )
    return {
        "load_seconds": load_seconds,
        "routing_checkpoint_seconds": checkpoint_seconds,
        "routing_checkpoint_epoch": checkpoint_epoch,
        "layout": layout_stats(partition_store),
        "queries": _measure_queries(
            _new_query_executor(partition_store, routing_store, index, config),
            queries,
            ground_truth,
            config,
            progress=progress,
            query_label="block",
        ),
    }


def _measure_queries(
    executor: QueryExecutor,
    queries: list[list[float]],
    ground_truth: np.ndarray,
    config: BenchmarkConfig,
    *,
    progress: ProgressCallback | None = None,
    query_label: str = "query",
) -> dict[str, dict[str, Any]]:
    measurements: dict[str, dict[str, Any]] = {}
    for fanout in config.routing_top_ks:
        label = "all" if fanout is None else str(fanout)
        _report(
            progress,
            f"{query_label} query started: fanout={label}",
        )
        fanout_started = time.perf_counter()

        for query in queries[:config.warmup_queries]:
            executor.query(
                {"g0": query},
                config.top_k,
                routing_top_k=fanout,
            )

        profiles: list[QueryProfile] = []
        predicted: list[list[int]] = []
        for query in queries:
            hits, profile = executor.query_profiled(
                {"g0": query},
                config.top_k,
                routing_top_k=fanout,
            )
            profiles.append(profile)
            predicted.append([int(hit.id) for hit in hits])

        latencies = [profile.total_seconds for profile in profiles]
        total = sum(latencies)
        measurements[label] = {
            "routing_top_k": label,
            "query_count": len(queries),
            "total_seconds": total,
            "mean_ms": 1000.0 * total / len(latencies),
            "p50_ms": 1000.0 * float(np.percentile(latencies, 50)),
            "p95_ms": 1000.0 * float(np.percentile(latencies, 95)),
            "qps": len(latencies) / total if total > 0.0 else float("inf"),
            "recall_at_k": recall_at_k(
                predicted,
                ground_truth,
                config.top_k,
            ),
            "mean_breakdown_ms": _aggregate_query_profiles(profiles),
            "p95_breakdown_ms": _aggregate_query_profiles(
                profiles,
                percentile=95,
            ),
            "phase_accounting": profiles[0].phase_accounting,
            "mean_selected_partitions": float(np.mean([
                profile.selected_partition_count for profile in profiles
            ])),
            "mean_loaded_segments": float(np.mean([
                profile.loaded_segment_count for profile in profiles
            ])),
        }
        _report(
            progress,
            f"{query_label} query completed: fanout={label} "
            f"({time.perf_counter() - fanout_started:.3f}s)",
        )
    return measurements


def _aggregate_query_profiles(
    profiles: list[QueryProfile],
    percentile: float | None = None,
) -> dict[str, float]:
    """Aggregate non-overlapping query phases as milliseconds."""

    rows = [
        {**profile.phase_seconds(), "other": profile.other_seconds}
        for profile in profiles
    ]
    return {
        phase: 1000.0 * (
            float(np.mean([row[phase] for row in rows]))
            if percentile is None
            else float(np.percentile(
                [row[phase] for row in rows],
                percentile,
            ))
        )
        for phase in rows[0]
    }


def compute_ground_truth(
    config: BenchmarkConfig,
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
) -> tuple[np.ndarray, str]:
    _, full_database_rows = _framed_vector_shape(
        config.dataset_dir / f"{config.dataset_prefix}_base.fvecs"
    )
    official_name = f"{config.dataset_prefix}_groundtruth.ivecs"
    official_path = config.dataset_dir / official_name
    use_official = (
        len(base_vectors) == full_database_rows
        and official_path.exists()
        and config.top_k <= 100
    )
    if use_official:
        official = read_ivecs(official_path, len(query_vectors))
        return official[:, :config.top_k], official_name

    exact = faiss.IndexFlatL2(base_vectors.shape[1])
    exact.add(np.ascontiguousarray(base_vectors, dtype=np.float32))
    _, neighbors = exact.search(
        np.ascontiguousarray(query_vectors, dtype=np.float32),
        config.top_k,
    )
    return neighbors.astype(np.int64, copy=False), "faiss.IndexFlatL2"


def layout_stats(partition_store: PartitionStore) -> dict[str, Any]:
    """Summarize visible vector counts without reading segment objects."""

    partition_ids = partition_store.list_partitions()
    manifests = [
        partition_store.read_current_manifest(partition_id)[0]
        for partition_id in partition_ids
    ]
    partition_vector_counts = {
        manifest.partition_id: (
            (manifest.base.count if manifest.base is not None else 0)
            + sum(delta.count for delta in manifest.visible_deltas)
        )
        for manifest in manifests
    }
    vector_counts = np.asarray(
        list(partition_vector_counts.values()),
        dtype=np.float64,
    )
    if vector_counts.size:
        mean = float(np.mean(vector_counts))
        partition_vector_stats = {
            "min": int(np.min(vector_counts)),
            "max": int(np.max(vector_counts)),
            "mean": mean,
            "stddev": float(np.std(vector_counts)),
            "p50": float(np.percentile(vector_counts, 50)),
            "p95": float(np.percentile(vector_counts, 95)),
            "imbalance_ratio": float(np.max(vector_counts) / mean),
        }
    else:
        partition_vector_stats = {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "stddev": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "imbalance_ratio": 0.0,
        }

    return {
        "partition_count": len(partition_ids),
        "base_records": sum(
            manifest.base.count if manifest.base is not None else 0
            for manifest in manifests
        ),
        "delta_segments": sum(
            len(manifest.visible_deltas)
            for manifest in manifests
        ),
        "delta_records": sum(
            delta.count
            for manifest in manifests
            for delta in manifest.visible_deltas
        ),
        "partition_vector_counts": partition_vector_counts,
        "partition_vector_stats": partition_vector_stats,
    }


def _publish_routing_checkpoint(
    routing_store: RoutingStore,
) -> tuple[float, int]:
    """Publish pending routing updates before measuring steady-state queries."""

    started = time.perf_counter()
    published = routing_store.publish_merged_checkpoint()
    elapsed = time.perf_counter() - started
    if not published:
        raise RuntimeError("routing checkpoint publication conflict in benchmark")
    checkpoint, _ = routing_store.read_current_checkpoint()
    return elapsed, checkpoint.epoch


def _system(
    root: Path,
    dataset_id: str,
    config: BenchmarkConfig,
) -> tuple[ANNIndex, PartitionStore, RoutingStore]:
    index = _index_backend(config)
    object_store = create_benchmark_store(
        config.storage_backend,
        root,
        config.storage_namespace,
        dataset_id,
    )
    return (
        index,
        PartitionStore(object_store, index, dataset_id),
        RoutingStore(object_store, dataset_id),
    )



def _new_query_executor(
    partition_store: PartitionStore,
    routing_store: RoutingStore,
    index: ANNIndex,
    config: BenchmarkConfig,
) -> QueryExecutor:
    """Create one query executor with benchmark-wide concurrency settings."""

    return QueryExecutor(
        partition_store,
        routing_store,
        index,
        partition_workers=config.query_partition_workers,
        object_read_workers=config.query_object_workers,
    )

def _index_backend(config: BenchmarkConfig) -> ANNIndex:
    if config.index_backend == "flat":
        return FlatIndex()
    return FaissHNSWIndex(
        m=config.hnsw_m,
        ef_construction=config.hnsw_ef_construction,
        ef_search=config.hnsw_ef_search,
    )


def _records(
    vectors: np.ndarray,
    indices: Iterable[int],
) -> list[VectorRecord]:
    return [
        VectorRecord(str(int(index)), vectors[int(index)].tolist())
        for index in indices
    ]


def _batches(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def _kmeans_policy(
    config: BenchmarkConfig,
    cluster_count: int,
) -> KMeansPartitionPolicy:
    return KMeansPartitionPolicy(
        n_clusters=cluster_count,
        random_state=config.seed,
        max_iter=config.ingest_max_iter,
        n_init=config.ingest_n_init,
    )


def _group_assigned_records(
    records: list[VectorRecord],
    assignments: list[str],
) -> dict[str, list[VectorRecord]]:
    grouped: dict[str, list[VectorRecord]] = {}
    for record, partition_id in zip(records, assignments, strict=True):
        grouped.setdefault(partition_id, []).append(record)
    return grouped


def _running_targets(
    partition_ids: list[str],
    counts: dict[str, int],
    sums: dict[str, np.ndarray],
) -> list[PartitionTarget]:
    return [
        PartitionTarget(
            partition_id=partition_id,
            centroid=(
                None
                if counts[partition_id] == 0
                else (sums[partition_id] / counts[partition_id]).tolist()
            ),
        )
        for partition_id in partition_ids
    ]


def _update_running_centroids(
    counts: dict[str, int],
    sums: dict[str, np.ndarray],
    records: list[VectorRecord],
    assignments: list[str],
) -> None:
    for record, partition_id in zip(records, assignments, strict=True):
        counts[partition_id] += 1
        sums[partition_id] += np.asarray(record.vector, dtype=np.float64)


def _framed_vector_shape(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        first = np.frombuffer(handle.read(4), dtype="<i4")
    if len(first) != 1 or int(first[0]) <= 0:
        raise ValueError(f"invalid framed vector file: {path}")
    dimension = int(first[0])
    row_bytes = 4 * (dimension + 1)
    size = path.stat().st_size
    if size % row_bytes != 0:
        raise ValueError(f"truncated framed vector file: {path}")
    return dimension, size // row_bytes


def _report(
    progress: ProgressCallback | None,
    message: str,
) -> None:
    if progress is not None:
        progress(message)


def _format_elapsed(seconds: float) -> str:
    """Format non-negative elapsed seconds as HH:MM:SS.mmm."""

    total_milliseconds = max(0, int(seconds * 1000.0))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _print_progress(started: float, message: str) -> None:
    """Print one flushed CLI progress event with total elapsed time."""

    elapsed = _format_elapsed(time.perf_counter() - started)
    print(f"[elapsed {elapsed}] {message}", flush=True)


def _json_config(config: BenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_dir"] = str(config.dataset_dir)
    payload["work_parent"] = (
        str(config.work_parent)
        if config.work_parent is not None
        else None
    )
    payload["routing_top_ks"] = [
        "all" if value is None else value
        for value in config.routing_top_ks
    ]
    return payload


def _parse_routing_top_ks(value: str) -> tuple[int | None, ...]:
    parsed: list[int | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        parsed.append(None if token == "all" else int(token))
    if not parsed:
        raise argparse.ArgumentTypeError("routing-top-k list must not be empty")
    return tuple(parsed)


def _print_summary(result: dict[str, Any]) -> None:
    dataset = result["dataset"]
    print(
        f"dataset: {dataset['database_vectors']} x {dataset['dimension']}, "
        f"queries={dataset['query_vectors']}, "
        f"read={dataset['read_seconds']:.3f}s, "
        f"ground_truth={dataset['ground_truth_source']}"
    )
    config = result["config"]
    print(
        "query workers: "
        f"partitions={config['query_partition_workers']}, "
        f"object_reads={config['query_object_workers']}"
    )
    print(
        "scenario                 fanout   load_s  compact_s route_cp_s  "
        "mean_ms   p95_ms      qps  recall"
    )

    scenarios = result["scenarios"]
    rows = [
        (
            "current-uncompacted",
            scenarios["current"]["load_seconds"],
            0.0,
            scenarios["current"]["uncompacted"]["routing_checkpoint_seconds"],
            scenarios["current"]["uncompacted"]["queries"],
        ),
        (
            "batch-local-uncompacted",
            scenarios["batch_local_semantic"]["load_seconds"],
            0.0,
            scenarios["batch_local_semantic"]["uncompacted"][
                "routing_checkpoint_seconds"
            ],
            scenarios["batch_local_semantic"]["uncompacted"]["queries"],
        ),
        (
            "global-clustering",
            scenarios["global_clustering"]["load_seconds"],
            0.0,
            scenarios["global_clustering"]["routing_checkpoint_seconds"],
            scenarios["global_clustering"]["queries"],
        ),
        (
            "block",
            scenarios["block"]["load_seconds"],
            0.0,
            scenarios["block"]["routing_checkpoint_seconds"],
            scenarios["block"]["queries"],
        ),
    ]
    if "compacted" in scenarios["current"]:
        rows.insert(1, (
            "current-compacted",
            scenarios["current"]["load_seconds"],
            scenarios["current"]["compaction_seconds"],
            scenarios["current"]["compacted"]["routing_checkpoint_seconds"],
            scenarios["current"]["compacted"]["queries"],
        ))
    if "compacted" in scenarios["batch_local_semantic"]:
        batch_local_position = next(
            index
            for index, row in enumerate(rows)
            if row[0] == "batch-local-uncompacted"
        )
        rows.insert(batch_local_position + 1, (
            "batch-local-compacted",
            scenarios["batch_local_semantic"]["load_seconds"],
            scenarios["batch_local_semantic"]["compaction_seconds"],
            scenarios["batch_local_semantic"]["compacted"][
                "routing_checkpoint_seconds"
            ],
            scenarios["batch_local_semantic"]["compacted"]["queries"],
        ))
    for (
        name,
        load_seconds,
        compact_seconds,
        checkpoint_seconds,
        measurements,
    ) in rows:
        for fanout, metrics in measurements.items():
            print(
                f"{name:24} {fanout:>6} "
                f"{load_seconds:8.3f} {compact_seconds:10.3f} "
                f"{checkpoint_seconds:10.3f} "
                f"{metrics['mean_ms']:8.3f} {metrics['p95_ms']:8.3f} "
                f"{metrics['qps']:8.2f} {metrics['recall_at_k']:7.4f}"
            )

    print()
    print("mean query phase work (ms; worker-seconds when partition-parallel)")
    print(
        "scenario                 fanout  parts   segs route_rd route_cpu "
        "manifest tombstone      ids index_rd idx_load      ann  visible "
        "global    other"
    )
    for name, _, _, _, measurements in rows:
        for fanout, metrics in measurements.items():
            phases = metrics["mean_breakdown_ms"]
            print(
                f"{name:24} {fanout:>6} "
                f"{metrics['mean_selected_partitions']:6.1f} "
                f"{metrics['mean_loaded_segments']:6.1f} "
                f"{phases['routing_read']:8.3f} "
                f"{phases['routing_compute']:9.3f} "
                f"{phases['manifest_read']:8.3f} "
                f"{phases['tombstone_read']:9.3f} "
                f"{phases['ids_read']:8.3f} "
                f"{phases['index_read']:8.3f} "
                f"{phases['index_deserialize']:8.3f} "
                f"{phases['ann_search']:8.3f} "
                f"{phases['visibility_merge']:8.3f} "
                f"{phases['global_merge']:7.3f} {phases['other']:8.3f}"
            )


def _parser(
    default_dataset_dir: Path = Path("/home/yangguo/Dataset/sift"),
    dataset_label: str = "SIFT1M",
    description: str | None = __doc__,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
    )
    parser.add_argument(
        "--database-limit",
        type=int,
        default=10_000,
        help=(
            "number of base vectors; 0 loads the complete "
            f"{dataset_label} dataset"
        ),
    )
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--partition-count", type=int, default=16)
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--ingest-batch-size", type=int, default=1_000)
    parser.add_argument("--ingest-clusters", type=int, default=16)
    parser.add_argument("--ingest-max-iter", type=int, default=100)
    parser.add_argument("--ingest-n-init", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--routing-top-k",
        type=_parse_routing_top_ks,
        default=(1, 4, 8, None),
        help="comma-separated partition fan-outs, for example 1,4,8,all",
    )
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument(
        "--compact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faiss-niter", type=int, default=25)
    parser.add_argument("--faiss-nredo", type=int, default=1)
    parser.add_argument(
        "--index-backend",
        choices=("flat", "faiss-hnsw"),
        default="faiss-hnsw",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--query-partition-workers", type=int, default=8)
    parser.add_argument("--query-object-workers", type=int, default=16)
    parser.add_argument("--work-parent", type=Path)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument(
        "--storage-backend",
        choices=SUPPORTED_STORAGE_BACKENDS,
        default="local",
    )
    parser.add_argument("--storage-namespace")
    parser.add_argument("--results-json", type=Path)
    return parser


def run_cli(
    parser: argparse.ArgumentParser,
    dataset_prefix: str,
) -> None:
    """Run one dataset-specific CLI and optionally persist its results."""

    args = parser.parse_args()
    config = BenchmarkConfig(
        dataset_dir=args.dataset_dir,
        dataset_prefix=dataset_prefix,
        database_limit=(
            None if args.database_limit == 0 else args.database_limit
        ),
        query_limit=args.query_limit,
        partition_count=args.partition_count,
        sample_fraction=args.sample_fraction,
        ingest_batch_size=args.ingest_batch_size,
        ingest_clusters=args.ingest_clusters,
        ingest_max_iter=args.ingest_max_iter,
        ingest_n_init=args.ingest_n_init,
        top_k=args.top_k,
        routing_top_ks=args.routing_top_k,
        warmup_queries=args.warmup_queries,
        compact=args.compact,
        seed=args.seed,
        faiss_niter=args.faiss_niter,
        faiss_nredo=args.faiss_nredo,
        index_backend=args.index_backend,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        query_partition_workers=args.query_partition_workers,
        query_object_workers=args.query_object_workers,
        storage_backend=args.storage_backend,
        storage_namespace=args.storage_namespace,
        work_parent=args.work_parent,
        keep_data=args.keep_data,
    )
    benchmark_started = time.perf_counter()
    result = run_benchmark(
        config,
        progress=lambda message: _print_progress(benchmark_started, message),
    )
    _print_summary(result)
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"results: {args.results_json}")


def main() -> None:
    """Run the SIFT CLI benchmark."""

    run_cli(_parser(), "sift")


if __name__ == "__main__":
    main()
