"""Benchmark global full rebuilds against partition-level generation migration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import multiprocessing
from pathlib import Path
from queue import Empty
import shutil
import tempfile
import threading
import time
from typing import Any, Callable

import faiss
import numpy as np

from benchmarks.storage import (
    SUPPORTED_STORAGE_BACKENDS,
    benchmark_storage_bytes,
    clear_benchmark_store,
    clone_benchmark_store,
    create_benchmark_store,
    describe_benchmark_storage,
    resolve_storage_namespace,
)
from benchmarks.sift_benchmark import (
    BenchmarkConfig,
    build_current_layout,
    compact_current_layout,
    layout_stats,
    read_fvecs,
)
from vela.execution.generation import GenerationTransformer
from vela.execution.migration import MigrationExecutor
from vela.index.faiss_hnsw import FaissHNSWIndex
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.protocol.commit import publish_pointer
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.storage.vector_codec import encode_float32_vectors
from vela.utils.ids import new_commit_id
from vela.utils.json_codec import to_json_bytes


@dataclass(frozen=True)
class EvolutionBenchmarkConfig:
    """Configuration for full rebuild and partition migration experiments."""

    dataset_dir: Path = Path("/home/yangguo/Dataset/sift")
    database_limit: int | None = 10_000
    partition_count: int = 16
    sample_fraction: float = 0.1
    ingest_batch_size: int = 1_000
    ingest_clusters: int = 16
    ingest_max_iter: int = 100
    ingest_n_init: int = 1
    seed: int = 0
    faiss_niter: int = 25
    faiss_nredo: int = 1
    source_index_backend: str = "faiss-hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    migration_workers: tuple[int, ...] = (1, 2, 4, 8)
    faiss_thread_budget: int = 16
    ivf_nlist: int = 128
    ivf_niter: int = 25
    full_indexes: tuple[str, ...] = ("ivf", "hnsw")
    migration_milestones: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    resource_sample_interval: float = 0.05
    storage_backend: str = "local"
    storage_namespace: str | None = None
    work_parent: Path | None = None
    keep_data: bool = False

    def layout_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            dataset_dir=self.dataset_dir,
            database_limit=self.database_limit,
            partition_count=self.partition_count,
            sample_fraction=self.sample_fraction,
            ingest_batch_size=self.ingest_batch_size,
            ingest_clusters=self.ingest_clusters,
            ingest_max_iter=self.ingest_max_iter,
            ingest_n_init=self.ingest_n_init,
            seed=self.seed,
            faiss_niter=self.faiss_niter,
            faiss_nredo=self.faiss_nredo,
            index_backend=self.source_index_backend,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            storage_backend=self.storage_backend,
            storage_namespace=self.storage_namespace,
            work_parent=self.work_parent,
            keep_data=self.keep_data,
        )

    def validate(self) -> None:
        self.layout_config().validate()
        if not self.full_indexes or any(
            name not in {"ivf", "hnsw"} for name in self.full_indexes
        ):
            raise ValueError("full_indexes must contain ivf and/or hnsw")
        if len(set(self.full_indexes)) != len(self.full_indexes):
            raise ValueError("full_indexes must be unique")
        if (
            not self.migration_workers
            or any(value < 1 for value in self.migration_workers)
            or tuple(sorted(set(self.migration_workers))) != self.migration_workers
        ):
            raise ValueError(
                "migration_workers must be unique, positive, and sorted"
            )
        if self.faiss_thread_budget < 1:
            raise ValueError("faiss_thread_budget must be positive")
        if max(self.migration_workers) > self.faiss_thread_budget:
            raise ValueError(
                "migration_workers must not exceed faiss_thread_budget"
            )
        if self.ivf_nlist < 1 or self.ivf_niter < 1:
            raise ValueError("IVF build settings must be positive")
        if (
            not self.migration_milestones
            or self.migration_milestones[-1] != 1.0
            or not {0.25, 0.5, 1.0}.issubset(self.migration_milestones)
            or any(
                value <= 0.0 or value > 1.0
                for value in self.migration_milestones
            )
            or tuple(sorted(set(self.migration_milestones)))
            != self.migration_milestones
        ):
            raise ValueError("migration milestones must be unique, sorted, and end at 1.0")
        if self.resource_sample_interval <= 0.0:
            raise ValueError("resource_sample_interval must be positive")


class IdentityTransformer(GenerationTransformer):
    """Copy vectors unchanged while publishing a new generation."""

    def transform(
        self,
        record: VectorRecord,
        target_generation: str,
    ) -> VectorRecord:
        return VectorRecord(
            id=record.id,
            vector=list(record.vector),
            metadata=dict(record.metadata),
            text=record.text,
        )


class ResourceSampler:
    """Sample process and directory resource peaks in one isolated worker."""

    def __init__(self, roots: list[Path], interval: float) -> None:
        self.roots = roots
        self.interval = interval
        self.baseline_storage_bytes = _directory_bytes(roots)
        self.baseline_rss_bytes = _process_status()[0]
        self.peak_storage_bytes = self.baseline_storage_bytes
        self.peak_rss_bytes = self.baseline_rss_bytes
        self.peak_cpu_percent = 0.0
        self.peak_thread_count = _process_status()[1]
        self._stop = threading.Event()
        self._started_wall = 0.0
        self._started_cpu = 0.0
        self._last_wall = 0.0
        self._last_cpu = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._started_wall = self._last_wall = time.perf_counter()
        self._started_cpu = self._last_cpu = time.process_time()
        self._thread.start()

    def sample(self) -> None:
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        elapsed = now_wall - self._last_wall
        if elapsed > 0.0:
            self.peak_cpu_percent = max(
                self.peak_cpu_percent,
                100.0 * (now_cpu - self._last_cpu) / elapsed,
            )
        self._last_wall = now_wall
        self._last_cpu = now_cpu
        rss, threads = _process_status()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.peak_thread_count = max(self.peak_thread_count, threads)
        self.peak_storage_bytes = max(
            self.peak_storage_bytes,
            _directory_bytes(self.roots),
        )

    def stop(self) -> dict[str, float | int]:
        self.sample()
        self._stop.set()
        self._thread.join()
        wall = time.perf_counter() - self._started_wall
        cpu = time.process_time() - self._started_cpu
        final_storage = _directory_bytes(self.roots)
        return {
            "baseline_storage_bytes": self.baseline_storage_bytes,
            "peak_storage_bytes": self.peak_storage_bytes,
            "peak_temporary_storage_bytes": max(
                0,
                self.peak_storage_bytes - self.baseline_storage_bytes,
            ),
            "final_storage_bytes": final_storage,
            "storage_amplification": (
                self.peak_storage_bytes / self.baseline_storage_bytes
                if self.baseline_storage_bytes
                else 0.0
            ),
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_incremental_rss_bytes": max(
                0,
                self.peak_rss_bytes - self.baseline_rss_bytes,
            ),
            "cpu_core_seconds": cpu,
            "mean_cpu_percent": 100.0 * cpu / wall if wall else 0.0,
            "peak_cpu_percent": self.peak_cpu_percent,
            "peak_thread_count": self.peak_thread_count,
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()


def run_evolution_benchmark(
    config: EvolutionBenchmarkConfig,
) -> dict[str, Any]:
    """Build one g0 source and run isolated full and partition migration workers."""

    config = replace(
        config,
        storage_namespace=resolve_storage_namespace(
            config.storage_backend,
            config.storage_namespace,
            "sift-evolution",
        ),
    )
    config.validate()
    layout_config = config.layout_config()
    dataset_started = time.perf_counter()
    base_vectors = read_fvecs(
        config.dataset_dir / "sift_base.fvecs",
        config.database_limit,
    )
    dataset_seconds = time.perf_counter() - dataset_started
    if config.partition_count > len(base_vectors):
        raise ValueError("partition_count must not exceed loaded database size")

    root = Path(tempfile.mkdtemp(
        prefix="vela-sift-evolution-",
        dir=str(config.work_parent) if config.work_parent else None,
    ))
    source_root = root / "source"
    setup_started = time.perf_counter()
    layout = build_current_layout(
        source_root,
        base_vectors,
        layout_config,
        dataset_id="evolution-source",
        generation="g0",
    )
    setup_compaction = compact_current_layout(layout)
    setup_seconds = time.perf_counter() - setup_started
    source_bytes = benchmark_storage_bytes(
        layout.partition_store.store,
        source_root,
    )

    result: dict[str, Any] = {
        "config": _json_config(config),
        "dataset": {
            "database_vectors": len(base_vectors),
            "dimension": int(base_vectors.shape[1]),
            "read_seconds": dataset_seconds,
        },
        "storage": describe_benchmark_storage(
            config.storage_backend,
            config.storage_namespace,
        ),
        "source_setup": {
            "seconds": setup_seconds,
            "compaction_seconds": setup_compaction,
            "storage_bytes": source_bytes,
            "layout": layout_stats(layout.partition_store),
        },
        "work_directory": str(root) if config.keep_data else None,
        "strategies": {},
    }
    try:
        for index_type in config.full_indexes:
            output_root = root / f"full-{index_type}"
            result["strategies"][f"full-{index_type}"] = _run_worker(
                _full_rebuild_worker,
                source_root,
                output_root,
                config,
                index_type,
            )

        partition_runs = [
            (name, worker_count, target)
            for worker_count in config.migration_workers
            for name, target in (
                ("partition-migration", _partition_migration_worker),
                ("partition-global-publish", _global_publication_worker),
            )
        ]
        for name, worker_count, target in partition_runs:
            strategy = f"{name}-w{worker_count}"
            migration_root = root / strategy
            destination_store = create_benchmark_store(
                config.storage_backend,
                migration_root,
                config.storage_namespace,
                strategy,
            )
            clone_benchmark_store(
                layout.partition_store.store,
                destination_store,
                source_root,
                migration_root,
            )
            result["strategies"][strategy] = _run_worker(
                target,
                migration_root,
                migration_root,
                config,
                worker_count,
            )
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


def _run_worker(
    target: Callable[..., None],
    source_root: Path,
    output_root: Path,
    config: EvolutionBenchmarkConfig,
    strategy: str | int | None,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=target,
        args=(
            str(source_root),
            str(output_root),
            config,
            strategy,
            queue,
        ),
    )
    process.start()
    while True:
        try:
            payload = queue.get(timeout=0.1)
            break
        except Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    f"evolution worker failed with exit code {process.exitcode}"
                )
    process.join()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    if process.exitcode != 0:
        raise RuntimeError(f"evolution worker failed with exit code {process.exitcode}")
    return payload


def _full_rebuild_worker(
    source_root_raw: str,
    output_root_raw: str,
    config: EvolutionBenchmarkConfig,
    index_type: str,
    queue: Any,
) -> None:
    try:
        faiss.omp_set_num_threads(config.faiss_thread_budget)
        source_root = Path(source_root_raw)
        output_root = Path(output_root_raw)
        output_store = create_benchmark_store(
            config.storage_backend,
            output_root,
            config.storage_namespace,
            f"full-{index_type}",
        )
        partition_store, partition_ids = _open_source(
            source_root,
            config,
            "evolution-source",
        )
        source_storage_bytes = benchmark_storage_bytes(
            partition_store.store, source_root)
        sampler = ResourceSampler(
            [source_root, output_root],
            config.resource_sample_interval,
        )
        sampler.start()
        total_started = time.perf_counter()

        read_started = time.perf_counter()
        records, read_milestones = _read_all_records(
            partition_store,
            partition_ids,
        )
        source_read_seconds = time.perf_counter() - read_started
        transform_started = time.perf_counter()
        transformed = [
            IdentityTransformer().transform(record, "g1")
            for record in records
        ]
        transform_seconds = time.perf_counter() - transform_started
        matrix = np.ascontiguousarray(
            [record.vector for record in transformed],
            dtype=np.float32,
        )
        ids = [record.id for record in transformed]

        train_seconds = 0.0
        build_started = time.perf_counter()
        if index_type == "ivf":
            nlist = min(config.ivf_nlist, len(matrix))
            quantizer = faiss.IndexFlatL2(matrix.shape[1])
            index = faiss.IndexIVFFlat(
                quantizer,
                matrix.shape[1],
                nlist,
                faiss.METRIC_L2,
            )
            index.cp.niter = config.ivf_niter
            index.cp.seed = config.seed
            train_started = time.perf_counter()
            index.train(matrix)
            train_seconds = time.perf_counter() - train_started
        else:
            index = faiss.IndexHNSWFlat(
                matrix.shape[1],
                config.hnsw_m,
                faiss.METRIC_L2,
            )
            index.hnsw.efConstruction = config.hnsw_ef_construction
        add_milestones = _add_with_milestones(index, matrix)
        index_build_seconds = time.perf_counter() - build_started - train_seconds

        serialize_started = time.perf_counter()
        vectors_blob = encode_float32_vectors(matrix)
        blob = faiss.serialize_index(index).tobytes()
        serialize_seconds = time.perf_counter() - serialize_started
        write_started = time.perf_counter()
        output_store.put("g1/index.faiss", blob, if_absent=True)
        output_store.put("g1/ids.json", to_json_bytes(ids), if_absent=True)
        output_store.put("g1/vectors.npy", vectors_blob, if_absent=True)
        output_store.put(
            "g1/generation.json",
            to_json_bytes({"generation": "g1", "index_type": index_type}),
            if_absent=True,
        )
        write_seconds = time.perf_counter() - write_started
        total_seconds = time.perf_counter() - total_started
        resources = sampler.stop()
        if config.storage_backend == "s3":
            final_storage_bytes = source_storage_bytes + benchmark_storage_bytes(
                output_store,
                output_root,
            )
            resources = _remote_storage_resources(
                resources, source_storage_bytes, final_storage_bytes)
        published = {
            "time_to_first_publishable_seconds": total_seconds,
            "time_to_25_percent_seconds": total_seconds,
            "time_to_50_percent_seconds": total_seconds,
            "time_to_100_percent_seconds": total_seconds,
        }
        queue.put({
            "strategy": f"full-{index_type}",
            "faiss_thread_budget": config.faiss_thread_budget,
            "faiss_threads_per_worker": config.faiss_thread_budget,
            "faiss_effective_thread_budget": config.faiss_thread_budget,
            "record_count": len(records),
            "source_read_seconds": source_read_seconds,
            "transform_seconds": transform_seconds,
            "train_seconds": train_seconds,
            "index_build_seconds": index_build_seconds,
            "serialize_seconds": serialize_seconds,
            "object_write_seconds": write_seconds,
            "total_seconds": total_seconds,
            "records_per_second": len(records) / total_seconds,
            "internal_read_milestones": read_milestones,
            "internal_index_milestones": add_milestones,
            "published_milestones": published,
            "index_bytes": len(blob),
            "resources": resources,
        })
    except Exception as error:
        queue.put({"error": f"{type(error).__name__}: {error}"})
        raise


def _global_publication_worker(
    source: str,
    output: str,
    config: EvolutionBenchmarkConfig,
    strategy: int,
    queue: Any,
) -> None:
    """Use identical migration tasks, with one collection publication barrier."""

    _partition_migration_worker(
        source, output, config, strategy, queue, global_publication=True,
    )


def _publish_collection_snapshot(
    partition_store: PartitionStore,
    partition_ids: list[str],
    details: list[dict[str, Any]],
) -> str:
    """Publish prepared manifests atomically for the quiescent benchmark source.

    This benchmark-only pointer is not consumed by production QueryExecutor.
    Partition current pointers remain on g0 throughout preparation and publication.
    """

    expected = set(partition_ids)
    if (
        len(details) != len(expected)
        or {item["partition_id"] for item in details} != expected
    ):
        raise ValueError("collection publication requires every partition")
    entries = {}
    for detail in details:
        current, _ = partition_store.read_current_manifest(detail["partition_id"])
        if (
            current.version != detail["source_version"]
            or current.active_generation != "g0"
        ):
            raise RuntimeError("source changed before collection publication")
        key = detail["prepared_manifest_key"]
        candidate = partition_store._read_manifest(key)
        if (
            candidate.partition_id != detail["partition_id"]
            or candidate.active_generation != "g1"
        ):
            raise ValueError("invalid prepared manifest")
        entries[detail["partition_id"]] = key
    prefix = f"datasets/{partition_store.dataset_id}/collection"
    key = f"{prefix}/snapshots/g1_{new_commit_id()}.json"
    partition_store.store.put(key, to_json_bytes({
        "generation": "g1", "manifests": entries,
        "routing_entries": {
            detail["partition_id"]: detail["prepared_routing_entry"]
            for detail in details
        },
    }), if_absent=True)
    if not publish_pointer(partition_store.store, f"{prefix}/generation.current", None, key):
        raise RuntimeError("collection publication conflict")
    return key


def _partition_migration_worker(
    source_root_raw: str,
    output_root_raw: str,
    config: EvolutionBenchmarkConfig,
    strategy: int,
    queue: Any,
    *,
    global_publication: bool = False,
) -> None:
    del output_root_raw
    try:
        worker_count = int(strategy)
        mode = "partition-global-publish" if global_publication else "partition-migration"
        strategy_name = f"{mode}-w{worker_count}"
        root = Path(source_root_raw)
        partition_store, partition_ids = _open_source(
            root,
            config,
            strategy_name,
        )
        routing_store = RoutingStore(
            partition_store.store,
            dataset_id="evolution-source",
        )
        counts = {}
        for partition_id in partition_ids:
            manifest, _ = partition_store.read_current_manifest(partition_id)
            counts[partition_id] = manifest.base.count if manifest.base else 0
        total_records = sum(counts.values())
        faiss_threads_per_worker = _faiss_threads_per_worker(
            config.faiss_thread_budget,
            worker_count,
        )
        ordered_partition_ids = _largest_partitions_first(
            partition_ids,
            counts,
        )

        sampler = ResourceSampler([root], config.resource_sample_interval)
        sampler.start()
        baseline_storage_bytes = benchmark_storage_bytes(
            partition_store.store, root)

        migration_started = time.perf_counter()
        migration = MigrationExecutor(partition_store, routing_store)
        migrated_records = 0
        first_seconds: float | None = None
        milestones: dict[str, Any] = {}
        details = []
        next_milestone = 0
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="partition-migration",
        ) as executor:
            futures = {
                executor.submit(
                    _migrate_partition,
                    migration,
                    partition_id,
                    faiss_threads_per_worker,
                    global_publication,
                ): partition_id
                for partition_id in ordered_partition_ids
            }
            for future in as_completed(futures):
                detail = future.result()
                active_seconds = time.perf_counter() - migration_started
                if first_seconds is None:
                    first_seconds = active_seconds
                migrated_records += detail["record_count"]
                fraction = migrated_records / total_records
                detail["completed_at_seconds"] = active_seconds
                detail["cumulative_record_fraction"] = fraction
                details.append(detail)
                while (
                    next_milestone < len(config.migration_milestones)
                    and fraction >= config.migration_milestones[next_milestone]
                ):
                    threshold = config.migration_milestones[next_milestone]
                    key = str(int(threshold * 100))
                    manifest_generations = _generation_counts(
                        partition_store,
                        partition_ids,
                    )
                    milestone = {
                        "target_fraction": threshold,
                        "elapsed_seconds": active_seconds,
                        "migrated_records": migrated_records,
                        "migrated_record_fraction": fraction,
                        "migrated_partitions": len(details),
                        "migrated_partition_fraction": (
                            len(details) / len(partition_ids)
                        ),
                        **manifest_generations,
                        "storage_bytes": benchmark_storage_bytes(
                            partition_store.store, root
                        ),
                    }
                    milestones[key] = milestone
                    next_milestone += 1

        preparation_seconds = time.perf_counter() - migration_started
        global_publish_seconds = 0.0
        snapshot_key = None
        if global_publication:
            publish_started = time.perf_counter()
            snapshot_key = _publish_collection_snapshot(partition_store, partition_ids, details)
            global_publish_seconds = time.perf_counter() - publish_started
        active_seconds = time.perf_counter() - migration_started
        resources = sampler.stop()
        durations = [item["duration_seconds"] for item in details]
        phase_metrics = _migration_phase_metrics(details)
        if config.storage_backend == "s3":
            final_storage_bytes = benchmark_storage_bytes(
                partition_store.store,
                root,
            )
            resources = _remote_storage_resources(
                resources, baseline_storage_bytes, final_storage_bytes)
        published_milestones = {
            "time_to_first_publishable_seconds": (
                active_seconds if global_publication else first_seconds
            ),
            **{
                f"time_to_{percent}_percent_seconds": (
                    active_seconds if global_publication else milestone["elapsed_seconds"]
                )
                for percent, milestone in milestones.items()
            },
        }
        queue.put({
            "strategy": strategy_name,
            "publication_mode": "global" if global_publication else "partition",
            "preparation_seconds": preparation_seconds,
            "global_publish_seconds": global_publish_seconds,
            "collection_snapshot_key": snapshot_key,
            "published_milestones": published_milestones,
            "preparation_milestones": milestones if global_publication else {},
            "time_to_first_prepared_partition_seconds": first_seconds if global_publication else None,
            "migration_workers": worker_count,
            "faiss_thread_budget": config.faiss_thread_budget,
            "faiss_threads_per_worker": faiss_threads_per_worker,
            "faiss_effective_thread_budget": (
                worker_count * faiss_threads_per_worker
            ),
            "scheduled_partition_ids": ordered_partition_ids,
            "record_count": total_records,
            "time_to_first_partition_seconds": published_milestones["time_to_first_publishable_seconds"],
            "time_to_25_percent_seconds": published_milestones["time_to_25_percent_seconds"],
            "time_to_50_percent_seconds": published_milestones["time_to_50_percent_seconds"],
            "time_to_100_percent_seconds": published_milestones["time_to_100_percent_seconds"],
            "total_seconds": active_seconds,
            "records_per_second": total_records / active_seconds,
            "partition_duration_mean_seconds": float(np.mean(durations)),
            "partition_duration_p50_seconds": float(np.percentile(durations, 50)),
            "partition_duration_p95_seconds": float(np.percentile(durations, 95)),
            "partition_duration_max_seconds": max(durations),
            **phase_metrics,
            "milestones": {} if global_publication else milestones,
            "partitions": details,
            "resources": resources,
        })
    except Exception as error:
        queue.put({"error": f"{type(error).__name__}: {error}"})
        raise


def _migrate_partition(
    migration: MigrationExecutor,
    partition_id: str,
    faiss_threads: int,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Migrate one independent partition and return task-local measurements."""

    faiss.omp_set_num_threads(faiss_threads)
    started = time.perf_counter()
    migration_result = migration.migrate_partition(
        partition_id,
        "g1",
        IdentityTransformer(),
        prepare_only=prepare_only,
    )
    duration = time.perf_counter() - started
    timings = migration_result.timings
    phase_seconds = {
        "read": timings.read_seconds,
        "transform": timings.transform_seconds,
        "index_build": timings.index_build_seconds,
        "serialize": timings.serialize_seconds,
        "write": timings.write_seconds,
        "publish": timings.publish_seconds,
    }
    instrumented_phase_seconds = sum(phase_seconds.values())
    if not migration_result.committed and not (prepare_only and migration_result.prepared_manifest_key):
        raise RuntimeError(
            f"partition migration did not commit: {partition_id}"
        )
    return {
        "partition_id": partition_id,
        "prepared_manifest_key": migration_result.prepared_manifest_key,
        "prepared_routing_entry": (
            asdict(migration_result.prepared_routing_entry)
            if migration_result.prepared_routing_entry is not None else None
        ),
        "faiss_threads": faiss_threads,
        "record_count": migration_result.live_count,
        "duration_seconds": duration,
        "records_per_second": (
            migration_result.live_count / duration
            if duration
            else 0.0
        ),
        "source_version": migration_result.source_version,
        "target_version": migration_result.new_version,
        "phase_seconds": phase_seconds,
        "instrumented_total_seconds": timings.total_seconds,
        "unattributed_seconds": max(
            0.0,
            timings.total_seconds - instrumented_phase_seconds,
        ),
    }


def _migration_phase_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate concurrent task phases as worker-seconds and distributions."""

    phases = tuple(details[0]["phase_seconds"])
    phase_values = {
        phase: [detail["phase_seconds"][phase] for detail in details]
        for phase in phases
    }
    return {
        "phase_worker_seconds": {
            phase: sum(values) for phase, values in phase_values.items()
        },
        "phase_mean_seconds": {
            phase: float(np.mean(values)) for phase, values in phase_values.items()
        },
        "phase_p95_seconds": {
            phase: float(np.percentile(values, 95))
            for phase, values in phase_values.items()
        },
        "phase_max_seconds": {
            phase: max(values) for phase, values in phase_values.items()
        },
    }


def _faiss_threads_per_worker(thread_budget: int, worker_count: int) -> int:
    """Allocate one fixed FAISS OpenMP share to every migration worker."""

    return max(1, thread_budget // worker_count)


def _largest_partitions_first(
    partition_ids: list[str],
    counts: dict[str, int],
) -> list[str]:
    """Schedule large partitions first, using ids as deterministic tie-breakers."""

    return sorted(
        partition_ids,
        key=lambda partition_id: (-counts[partition_id], partition_id),
    )


def _open_source(
    root: Path,
    config: EvolutionBenchmarkConfig,
    storage_scenario: str,
) -> tuple[PartitionStore, list[str]]:
    store = create_benchmark_store(
        config.storage_backend,
        root,
        config.storage_namespace,
        storage_scenario,
    )
    index = (
        FlatIndex()
        if config.source_index_backend == "flat"
        else FaissHNSWIndex(
            m=config.hnsw_m,
            ef_construction=config.hnsw_ef_construction,
        )
    )
    partition_store = PartitionStore(
        store,
        index,
        dataset_id="evolution-source",
    )
    return partition_store, partition_store.list_partitions()


def _read_all_records(
    partition_store: PartitionStore,
    partition_ids: list[str],
) -> tuple[list[VectorRecord], dict[str, float]]:
    counts = []
    manifests = []
    for partition_id in partition_ids:
        manifest, _ = partition_store.read_current_manifest(partition_id)
        manifests.append(manifest)
        counts.append(manifest.base.count if manifest.base else 0)
    total = sum(counts)
    records = []
    milestones: dict[str, float] = {}
    started = time.perf_counter()
    next_threshold = 0
    thresholds = (0.25, 0.5, 1.0)
    for manifest in manifests:
        records.extend(partition_store.materialize_live_records(manifest))
        fraction = len(records) / total
        while next_threshold < len(thresholds) and fraction >= thresholds[next_threshold]:
            milestones[str(int(thresholds[next_threshold] * 100))] = (
                time.perf_counter() - started
            )
            next_threshold += 1
    return records, milestones


def _add_with_milestones(index: Any, matrix: np.ndarray) -> dict[str, float]:
    thresholds = (0.25, 0.5, 1.0)
    milestones = {}
    started = time.perf_counter()
    offset = 0
    for threshold in thresholds:
        end = int(np.ceil(len(matrix) * threshold))
        if end > offset:
            index.add(matrix[offset:end])
            offset = end
        milestones[str(int(threshold * 100))] = time.perf_counter() - started
    return milestones


def _generation_counts(
    partition_store: PartitionStore,
    partition_ids: list[str],
) -> dict[str, int]:
    counts = {"g0_partition_count": 0, "g1_partition_count": 0}
    for partition_id in partition_ids:
        manifest, _ = partition_store.read_current_manifest(partition_id)
        counts[f"{manifest.active_generation}_partition_count"] += 1
    return counts

def _remote_storage_resources(
    resources: dict[str, float | int],
    baseline_bytes: int,
    final_bytes: int,
) -> dict[str, float | int]:
    """Replace local directory counters with monotonic S3 namespace sizes."""

    peak_bytes = max(baseline_bytes, final_bytes)
    updated = dict(resources)
    updated.update({
        "baseline_storage_bytes": baseline_bytes,
        "peak_storage_bytes": peak_bytes,
        "peak_temporary_storage_bytes": max(0, peak_bytes - baseline_bytes),
        "final_storage_bytes": final_bytes,
        "storage_amplification": (
            peak_bytes / baseline_bytes if baseline_bytes else 0.0
        ),
    })
    return updated



def _process_status() -> tuple[int, int]:
    rss = 0
    threads = 1
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("Threads:"):
                threads = int(line.split()[1])
    except FileNotFoundError:
        pass
    return rss, threads


def _directory_bytes(roots: list[Path]) -> int:
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def _json_config(config: EvolutionBenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_dir"] = str(config.dataset_dir)
    payload["work_parent"] = str(config.work_parent) if config.work_parent else None
    payload["full_indexes"] = list(config.full_indexes)
    payload["migration_workers"] = list(config.migration_workers)
    payload["migration_milestones"] = list(config.migration_milestones)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/yangguo/Dataset/sift"))
    parser.add_argument("--database-limit", type=int, default=10_000)
    parser.add_argument("--partition-count", type=int, default=16)
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--ingest-batch-size", type=int, default=1_000)
    parser.add_argument("--ingest-clusters", type=int, default=16)
    parser.add_argument("--ingest-max-iter", type=int, default=100)
    parser.add_argument("--ingest-n-init", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faiss-niter", type=int, default=25)
    parser.add_argument("--faiss-nredo", type=int, default=1)
    parser.add_argument("--full-indexes", type=lambda value: tuple(value.split(",")), default=("ivf", "hnsw"))
    parser.add_argument("--migration-workers", type=_parse_ints, default=(1, 2, 4, 8))
    parser.add_argument("--faiss-thread-budget", type=int, default=16)
    parser.add_argument("--ivf-nlist", type=int, default=128)
    parser.add_argument("--ivf-niter", type=int, default=25)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--source-index-backend", choices=("flat", "faiss-hnsw"), default="faiss-hnsw")
    parser.add_argument("--resource-sample-interval", type=float, default=0.05)
    parser.add_argument(
        "--storage-backend",
        choices=SUPPORTED_STORAGE_BACKENDS,
        default="local",
    )
    parser.add_argument("--storage-namespace")
    parser.add_argument("--work-parent", type=Path)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--results-json", type=Path)
    return parser


def _print_summary(result: dict[str, Any]) -> None:
    print("strategy                        first_s    t25_s    t50_s   total_s  peak_mem_mb  peak_tmp_mb")
    for name, strategy in result["strategies"].items():
        resources = strategy["resources"]
        published = strategy.get("published_milestones", {})
        first = strategy.get(
            "time_to_first_partition_seconds",
            published.get("time_to_first_publishable_seconds"),
        )
        t25 = strategy.get(
            "time_to_25_percent_seconds",
            published.get("time_to_25_percent_seconds"),
        )
        t50 = strategy.get(
            "time_to_50_percent_seconds",
            published.get("time_to_50_percent_seconds"),
        )
        print(
            f"{name:31} {first:8.3f} {t25:8.3f} {t50:8.3f} "
            f"{strategy['total_seconds']:9.3f} "
            f"{resources['peak_incremental_rss_bytes'] / 2**20:12.2f} "
            f"{resources['peak_temporary_storage_bytes'] / 2**20:11.2f}"
        )


    profiled = [
        (name, strategy)
        for name, strategy in result["strategies"].items()
        if "phase_worker_seconds" in strategy
    ]
    if not profiled:
        return
    print()
    print("migration worker-seconds    read transform    build serialize    write  publish")
    for name, strategy in profiled:
        phases = strategy["phase_worker_seconds"]
        print(
            f"{name:27} {phases['read']:7.3f} {phases['transform']:9.3f} "
            f"{phases['index_build']:8.3f} {phases['serialize']:9.3f} "
            f"{phases['write']:8.3f} {phases['publish']:8.3f}"
        )
def _parse_ints(value: str) -> tuple[int, ...]:
    """Parse one comma-separated integer list for argparse."""

    return tuple(int(item) for item in value.split(","))


def main() -> None:
    args = _parser().parse_args()
    config = EvolutionBenchmarkConfig(
        dataset_dir=args.dataset_dir,
        database_limit=None if args.database_limit == 0 else args.database_limit,
        partition_count=args.partition_count,
        sample_fraction=args.sample_fraction,
        ingest_batch_size=args.ingest_batch_size,
        ingest_clusters=args.ingest_clusters,
        ingest_max_iter=args.ingest_max_iter,
        ingest_n_init=args.ingest_n_init,
        seed=args.seed,
        faiss_niter=args.faiss_niter,
        source_index_backend=args.source_index_backend,
        faiss_nredo=args.faiss_nredo,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        ivf_nlist=args.ivf_nlist,
        ivf_niter=args.ivf_niter,
        full_indexes=args.full_indexes,
        migration_workers=args.migration_workers,
        faiss_thread_budget=args.faiss_thread_budget,
        resource_sample_interval=args.resource_sample_interval,
        storage_backend=args.storage_backend,
        storage_namespace=args.storage_namespace,
        work_parent=args.work_parent,
        keep_data=args.keep_data,
    )
    result = run_evolution_benchmark(config)
    _print_summary(result)
    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"results: {args.results_json}")


if __name__ == "__main__":
    main()
