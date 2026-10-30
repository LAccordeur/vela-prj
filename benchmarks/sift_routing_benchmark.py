"""Compare single- and multi-centroid routing on one compacted SIFT layout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import faiss
import numpy as np

from benchmarks.storage import (
    SUPPORTED_STORAGE_BACKENDS,
    clear_benchmark_store,
    create_benchmark_store,
    describe_benchmark_storage,
    resolve_storage_namespace,
)
from benchmarks.sift_benchmark import (
    BenchmarkConfig,
    build_current_layout,
    compact_current_layout,
    compute_ground_truth,
    layout_stats,
    read_fvecs,
    recall_at_k,
)
from vela.storage.partition_store import PartitionStore
from vela.execution.query import QueryExecutor


RoutingModel = dict[str, np.ndarray]


@dataclass(frozen=True)
class RoutingBenchmarkConfig:
    """Configuration for one compacted-layout routing comparison."""

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
    centroid_counts: tuple[int, ...] = (1, 2, 4, 8)
    warmup_queries: int = 3
    seed: int = 0
    faiss_niter: int = 25
    faiss_nredo: int = 1
    centroid_niter: int = 25
    centroid_nredo: int = 1
    index_backend: str = "faiss-hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    storage_backend: str = "local"
    storage_namespace: str | None = None
    work_parent: Path | None = None
    keep_data: bool = False

    def layout_config(self) -> BenchmarkConfig:
        """Return the shared current-layout build configuration."""

        return BenchmarkConfig(
            dataset_dir=self.dataset_dir,
            database_limit=self.database_limit,
            query_limit=self.query_limit,
            partition_count=self.partition_count,
            sample_fraction=self.sample_fraction,
            ingest_batch_size=self.ingest_batch_size,
            ingest_clusters=self.ingest_clusters,
            ingest_max_iter=self.ingest_max_iter,
            ingest_n_init=self.ingest_n_init,
            top_k=self.top_k,
            routing_top_ks=self.routing_top_ks,
            warmup_queries=self.warmup_queries,
            compact=True,
            seed=self.seed,
            faiss_niter=self.faiss_niter,
            faiss_nredo=self.faiss_nredo,
            index_backend=self.index_backend,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_ef_search=self.hnsw_ef_search,
            work_parent=self.work_parent,
            storage_backend=self.storage_backend,
            storage_namespace=self.storage_namespace,
            keep_data=self.keep_data,
        )

    def validate(self) -> None:
        """Reject invalid layout, centroid, and measurement settings."""

        self.layout_config().validate()
        if not self.centroid_counts:
            raise ValueError("at least one routing centroid count is required")
        if len(set(self.centroid_counts)) != len(self.centroid_counts):
            raise ValueError("routing centroid counts must be unique")
        if any(count < 1 for count in self.centroid_counts):
            raise ValueError("routing centroid counts must be positive")
        if self.centroid_niter < 1 or self.centroid_nredo < 1:
            raise ValueError("routing centroid K-Means settings must be positive")


def run_routing_benchmark(
    config: RoutingBenchmarkConfig,
) -> dict[str, Any]:
    """Build one compacted current layout and compare routing centroid counts."""

    config = replace(
        config,
        storage_namespace=resolve_storage_namespace(
            config.storage_backend,
            config.storage_namespace,
            "sift-routing",
        ),
    )
    config.validate()
    layout_config = config.layout_config()

    dataset_started = time.perf_counter()
    base_vectors = read_fvecs(
        config.dataset_dir / "sift_base.fvecs",
        config.database_limit,
    )
    query_vectors = read_fvecs(
        config.dataset_dir / "sift_query.fvecs",
        config.query_limit,
    )
    dataset_seconds = time.perf_counter() - dataset_started
    if config.partition_count > len(base_vectors):
        raise ValueError("partition_count must not exceed loaded database size")
    if config.top_k > len(base_vectors):
        raise ValueError("top_k must not exceed loaded database size")

    ground_truth_started = time.perf_counter()
    ground_truth, ground_truth_source = compute_ground_truth(
        layout_config,
        base_vectors,
        query_vectors,
    )
    ground_truth_seconds = time.perf_counter() - ground_truth_started

    work_parent = (
        str(config.work_parent)
        if config.work_parent is not None
        else None
    )
    root = Path(tempfile.mkdtemp(
        prefix="vela-sift-routing-",
        dir=work_parent,
    ))
    result: dict[str, Any] = {
        "config": _json_config(config),
        "dataset": {
            "database_vectors": len(base_vectors),
            "query_vectors": len(query_vectors),
            "dimension": int(base_vectors.shape[1]),
            "read_seconds": dataset_seconds,
            "ground_truth_seconds": ground_truth_seconds,
            "ground_truth_source": ground_truth_source,
        },
        "storage": describe_benchmark_storage(
            config.storage_backend,
            config.storage_namespace,
        ),
        "work_directory": str(root) if config.keep_data else None,
    }
    try:
        layout = build_current_layout(
            root / "current-compacted",
            base_vectors,
            layout_config,
            dataset_id="routing-benchmark",
        )
        compaction_seconds = compact_current_layout(layout)
        result["layout"] = {
            "sample_build_seconds": layout.sample_seconds,
            "ingest_seconds": layout.ingest_seconds,
            "load_seconds": layout.load_seconds,
            "compaction_seconds": compaction_seconds,
            "sample_records": layout.sample_count,
            "ingested_records": layout.ingested_count,
            "objects": layout_stats(layout.partition_store),
        }

        vector_started = time.perf_counter()
        partition_vectors = _load_partition_vectors(
            layout.partition_store,
            layout.partition_ids,
            int(base_vectors.shape[1]),
        )
        result["partition_vector_load_seconds"] = (
            time.perf_counter() - vector_started
        )

        executor = QueryExecutor(
            layout.partition_store,
            layout.routing_store,
            layout.index,
        )
        query_lists = [vector.tolist() for vector in query_vectors]
        strategies: dict[str, Any] = {}
        for centroid_count in config.centroid_counts:
            build_started = time.perf_counter()
            model = build_routing_model(
                layout.partition_store,
                layout.partition_ids,
                partition_vectors,
                centroid_count,
                config,
            )
            build_seconds = time.perf_counter() - build_started
            strategies[str(centroid_count)] = {
                "requested_centroids_per_partition": centroid_count,
                "total_centroids": sum(
                    len(centroids)
                    for centroids in model.values()
                ),
                "build_seconds": build_seconds,
                "measurements": measure_routing_strategy(
                    executor,
                    model,
                    query_lists,
                    ground_truth,
                    config,
                ),
            }
        result["strategies"] = strategies
        return result
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


def _load_partition_vectors(
    partition_store: PartitionStore,
    partition_ids: list[str],
    dimension: int,
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for partition_id in partition_ids:
        manifest, _ = partition_store.read_current_manifest(partition_id)
        records = partition_store.materialize_live_records(manifest)
        if records:
            matrix = np.ascontiguousarray(
                [record.vector for record in records],
                dtype=np.float32,
            )
        else:
            matrix = np.empty((0, dimension), dtype=np.float32)
        matrices[partition_id] = matrix
    return matrices


def build_routing_model(
    partition_store: PartitionStore,
    partition_ids: list[str],
    partition_vectors: dict[str, np.ndarray],
    centroid_count: int,
    config: RoutingBenchmarkConfig,
) -> RoutingModel:
    """Build one routing model over an already compacted partition layout."""

    if centroid_count < 1:
        raise ValueError("centroid_count must be positive")
    if centroid_count == 1:
        model: RoutingModel = {}
        for partition_id in partition_ids:
            manifest, _ = partition_store.read_current_manifest(partition_id)
            summary = partition_store.read_base_summary(manifest)
            centroid = summary.get("centroid")
            model[partition_id] = (
                np.empty((0, 0), dtype=np.float32)
                if centroid is None
                else np.ascontiguousarray([centroid], dtype=np.float32)
            )
        return model

    model = {}
    for partition_id in partition_ids:
        matrix = partition_vectors[partition_id]
        effective_count = min(centroid_count, len(matrix))
        if effective_count == 0:
            model[partition_id] = np.empty(
                (0, matrix.shape[1]),
                dtype=np.float32,
            )
            continue
        if effective_count == 1:
            model[partition_id] = np.mean(
                matrix,
                axis=0,
                keepdims=True,
                dtype=np.float32,
            )
            continue
        kmeans = faiss.Kmeans(
            matrix.shape[1],
            effective_count,
            niter=config.centroid_niter,
            nredo=config.centroid_nredo,
            seed=config.seed,
            verbose=False,
            gpu=False,
        )
        kmeans.cp.min_points_per_centroid = 1
        kmeans.train(matrix)
        model[partition_id] = np.ascontiguousarray(
            kmeans.centroids,
            dtype=np.float32,
        )
    return model


def select_partitions(
    query: list[float],
    model: RoutingModel,
    routing_top_k: int | None,
) -> list[str]:
    """Rank partitions by their nearest centroid using squared L2 distance."""

    if routing_top_k is not None and routing_top_k < 1:
        raise ValueError("routing_top_k must be positive or None")
    if routing_top_k is None:
        return sorted(model)

    query_vector = np.asarray(query, dtype=np.float32)
    scored: list[tuple[float, str]] = []
    for partition_id, centroids in model.items():
        if len(centroids) == 0:
            continue
        if centroids.shape[1] != len(query_vector):
            raise ValueError("query and routing centroids must have the same dimension")
        distances = np.sum(
            (centroids - query_vector) ** 2,
            axis=1,
        )
        scored.append((-float(np.min(distances)), partition_id))
    return [
        partition_id
        for _, partition_id
        in sorted(scored, reverse=True)[:routing_top_k]
    ]


def measure_routing_strategy(
    executor: QueryExecutor,
    model: RoutingModel,
    queries: list[list[float]],
    ground_truth: np.ndarray,
    config: RoutingBenchmarkConfig,
) -> dict[str, dict[str, float | int | str]]:
    """Measure routing-only, search-only, end-to-end latency, and recall."""

    measurements: dict[str, dict[str, float | int | str]] = {}
    for fanout in config.routing_top_ks:
        for query in queries[:config.warmup_queries]:
            selected = select_partitions(query, model, fanout)
            executor.query(
                {"g0": query},
                config.top_k,
                candidate_partitions=selected,
            )

        routing_latencies: list[float] = []
        search_latencies: list[float] = []
        total_latencies: list[float] = []
        selected_counts: list[int] = []
        predicted: list[list[int]] = []
        for query in queries:
            total_started = time.perf_counter()
            routing_started = time.perf_counter()
            selected = select_partitions(query, model, fanout)
            routing_elapsed = time.perf_counter() - routing_started
            search_started = time.perf_counter()
            hits = executor.query(
                {"g0": query},
                config.top_k,
                candidate_partitions=selected,
            )
            search_elapsed = time.perf_counter() - search_started
            total_elapsed = time.perf_counter() - total_started

            routing_latencies.append(routing_elapsed)
            search_latencies.append(search_elapsed)
            total_latencies.append(total_elapsed)
            selected_counts.append(len(selected))
            predicted.append([int(hit.id) for hit in hits])

        label = "all" if fanout is None else str(fanout)
        total = sum(total_latencies)
        measurements[label] = {
            "routing_top_k": label,
            "query_count": len(queries),
            "mean_selected_partitions": float(np.mean(selected_counts)),
            "routing_mean_ms": _mean_ms(routing_latencies),
            "routing_p50_ms": _percentile_ms(routing_latencies, 50),
            "routing_p95_ms": _percentile_ms(routing_latencies, 95),
            "search_mean_ms": _mean_ms(search_latencies),
            "search_p50_ms": _percentile_ms(search_latencies, 50),
            "search_p95_ms": _percentile_ms(search_latencies, 95),
            "total_mean_ms": _mean_ms(total_latencies),
            "total_p50_ms": _percentile_ms(total_latencies, 50),
            "total_p95_ms": _percentile_ms(total_latencies, 95),
            "qps": len(queries) / total if total > 0.0 else float("inf"),
            "recall_at_k": recall_at_k(
                predicted,
                ground_truth,
                config.top_k,
            ),
        }
    return measurements


def _mean_ms(values: list[float]) -> float:
    return 1000.0 * sum(values) / len(values)


def _percentile_ms(values: list[float], percentile: int) -> float:
    return 1000.0 * float(np.percentile(values, percentile))


def _json_config(config: RoutingBenchmarkConfig) -> dict[str, Any]:
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
    payload["centroid_counts"] = list(config.centroid_counts)
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


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError(
            "routing-centroids must contain positive integers"
        )
    return parsed


def _print_summary(result: dict[str, Any]) -> None:
    dataset = result["dataset"]
    layout = result["layout"]
    print(
        f"dataset: {dataset['database_vectors']} x {dataset['dimension']}, "
        f"queries={dataset['query_vectors']}, "
        f"load={layout['load_seconds']:.3f}s, "
        f"compact={layout['compaction_seconds']:.3f}s"
    )
    print(
        "centroids fanout  build_s route_ms search_ms total_ms "
        "p95_ms      qps  recall"
    )
    for strategy in result["strategies"].values():
        count = strategy["requested_centroids_per_partition"]
        for fanout, metrics in strategy["measurements"].items():
            print(
                f"{count:9d} {fanout:>6} "
                f"{strategy['build_seconds']:8.3f} "
                f"{metrics['routing_mean_ms']:8.3f} "
                f"{metrics['search_mean_ms']:9.3f} "
                f"{metrics['total_mean_ms']:8.3f} "
                f"{metrics['total_p95_ms']:8.3f} "
                f"{metrics['qps']:8.2f} "
                f"{metrics['recall_at_k']:7.4f}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/yangguo/Dataset/sift"),
    )
    parser.add_argument("--database-limit", type=int, default=10_000)
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
    )
    parser.add_argument(
        "--routing-centroids",
        type=_parse_positive_ints,
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faiss-niter", type=int, default=25)
    parser.add_argument("--faiss-nredo", type=int, default=1)
    parser.add_argument("--centroid-niter", type=int, default=25)
    parser.add_argument("--centroid-nredo", type=int, default=1)
    parser.add_argument(
        "--index-backend",
        choices=("flat", "faiss-hnsw"),
        default="faiss-hnsw",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--work-parent", type=Path)
    parser.add_argument(
        "--storage-backend",
        choices=SUPPORTED_STORAGE_BACKENDS,
        default="local",
    )
    parser.add_argument("--storage-namespace")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--results-json", type=Path)
    return parser


def main() -> None:
    """Run the routing benchmark and optionally persist structured results."""

    args = _parser().parse_args()
    config = RoutingBenchmarkConfig(
        dataset_dir=args.dataset_dir,
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
        centroid_counts=args.routing_centroids,
        warmup_queries=args.warmup_queries,
        seed=args.seed,
        faiss_niter=args.faiss_niter,
        faiss_nredo=args.faiss_nredo,
        centroid_niter=args.centroid_niter,
        centroid_nredo=args.centroid_nredo,
        index_backend=args.index_backend,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        storage_backend=args.storage_backend,
        storage_namespace=args.storage_namespace,
        work_parent=args.work_parent,
        keep_data=args.keep_data,
    )
    result = run_routing_benchmark(config)
    _print_summary(result)
    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"results: {args.results_json}")


if __name__ == "__main__":
    main()

