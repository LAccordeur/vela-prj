"""Execution-layer query processing."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import math
from threading import BoundedSemaphore
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from vela.index.base import ANNIndex, LoadedANNIndex, LocalIndexHit
from vela.model.manifest import DeltaRef, PartitionManifest, SegmentRef
from vela.model.record import SearchHit
from vela.storage.partition_store import PartitionStore, ReferencedObjectMissing
from vela.storage.routing_store import RoutingStore

T = TypeVar("T")


@dataclass
class QueryProfile:
    """Timing profile produced by `QueryExecutor.query_profiled`."""

    routing_read_seconds: float = 0.0
    routing_compute_seconds: float = 0.0
    manifest_read_seconds: float = 0.0
    tombstone_read_seconds: float = 0.0
    ids_read_seconds: float = 0.0
    index_read_seconds: float = 0.0
    index_deserialize_seconds: float = 0.0
    ann_search_seconds: float = 0.0
    visibility_merge_seconds: float = 0.0
    global_merge_seconds: float = 0.0
    total_seconds: float = 0.0
    selected_partition_count: int = 0
    loaded_segment_count: int = 0
    phase_accounting: str = "wall"

    def phase_seconds(self) -> dict[str, float]:
        """Return measured wall phases or parallel worker-seconds."""

        return {
            "routing_read": self.routing_read_seconds,
            "routing_compute": self.routing_compute_seconds,
            "manifest_read": self.manifest_read_seconds,
            "tombstone_read": self.tombstone_read_seconds,
            "ids_read": self.ids_read_seconds,
            "index_read": self.index_read_seconds,
            "index_deserialize": self.index_deserialize_seconds,
            "ann_search": self.ann_search_seconds,
            "visibility_merge": self.visibility_merge_seconds,
            "global_merge": self.global_merge_seconds,
        }

    @property
    def other_seconds(self) -> float:
        """Return unassigned orchestration and profiling overhead."""

        return max(
            0.0,
            self.total_seconds - sum(self.phase_seconds().values()),
        )

    def add_parallel_work(self, other: "QueryProfile") -> None:
        """Add one partition task's phase work to this query profile."""

        for name in self.phase_seconds():
            attribute = f"{name}_seconds"
            value = getattr(self, attribute) + getattr(other, attribute)
            setattr(self, attribute, value)
        self.loaded_segment_count += other.loaded_segment_count
        self.phase_accounting = "worker-seconds"


class _BoundedObjectReadPool:
    """Executor-wide object-read pool with bounded running and queued work."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="vela-object-read",
            )
            if max_workers > 1
            else None
        )
        self._slots = BoundedSemaphore(max_workers * 2)

    def run(self, operations: list[Callable[[], T]]) -> list[T]:
        if self._executor is None:
            return [operation() for operation in operations]
        futures = [self._submit(operation) for operation in operations]
        return [future.result() for future in futures]

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _submit(self, operation: Callable[[], T]) -> Future[T]:
        self._slots.acquire()
        try:
            future = self._executor.submit(operation)  # type: ignore[union-attr]
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future


class QueryExecutor:
    """Searches partitions by following committed manifests."""

    def __init__(
        self,
        partition_store: PartitionStore,
        routing_store: RoutingStore,
        index: ANNIndex,
        max_missing_retries: int = 2,
        routing_metric: str = "l2",
        partition_workers: int = 1,
        object_read_workers: int = 1,
    ) -> None:
        if routing_metric not in {"l2", "cosine"}:
            raise ValueError(f"unsupported routing metric: {routing_metric}")
        if partition_workers < 1 or object_read_workers < 1:
            raise ValueError(
                "query partition and object-read workers must be positive"
            )
        self.partition_store = partition_store
        self.routing_store = routing_store
        self.index = index
        self.max_missing_retries = max_missing_retries
        self.routing_metric = routing_metric
        self.partition_workers = partition_workers
        self.object_read_workers = object_read_workers
        self._partition_pool = (
            ThreadPoolExecutor(
                max_workers=partition_workers,
                thread_name_prefix="vela-query-partition",
            )
            if partition_workers > 1
            else None
        )
        self._object_reads = _BoundedObjectReadPool(object_read_workers)

    def close(self) -> None:
        """Wait for outstanding query work and release worker threads."""

        if self._partition_pool is not None:
            self._partition_pool.shutdown(wait=True)
        self._object_reads.close()

    def __enter__(self) -> "QueryExecutor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def query(
        self,
        query_vector_by_generation: dict[str, list[float]],
        top_k: int,
        candidate_partitions: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        routing_top_k: int | None = None,
    ) -> list[SearchHit]:
        """Search candidate partitions and merge local results."""

        return self._query(
            query_vector_by_generation,
            top_k,
            candidate_partitions,
            metadata_filter,
            routing_top_k,
            profile=None,
        )

    def query_profiled(
        self,
        query_vector_by_generation: dict[str, list[float]],
        top_k: int,
        candidate_partitions: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        routing_top_k: int | None = None,
    ) -> tuple[list[SearchHit], QueryProfile]:
        """Search and return end-to-end wall time plus phase work."""

        profile = QueryProfile()
        started = time.perf_counter()
        hits = self._query(
            query_vector_by_generation,
            top_k,
            candidate_partitions,
            metadata_filter,
            routing_top_k,
            profile=profile,
        )
        profile.total_seconds = time.perf_counter() - started
        return hits, profile

    def _query(
        self,
        query_vector_by_generation: dict[str, list[float]],
        top_k: int,
        candidate_partitions: list[str] | None,
        metadata_filter: dict[str, Any] | None,
        routing_top_k: int | None,
        profile: QueryProfile | None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        partitions = (
            candidate_partitions
            if candidate_partitions is not None
            else self._candidate_partitions(
                query_vector_by_generation,
                metadata_filter,
                routing_top_k,
                profile,
            )
        )
        if profile is not None:
            profile.selected_partition_count = len(partitions)
        hits = self._search_partitions(
            partitions,
            query_vector_by_generation,
            top_k,
            visited=set(),
            profile=profile,
        )
        started = time.perf_counter() if profile is not None else 0.0
        ranked = sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
        if profile is not None:
            profile.global_merge_seconds += time.perf_counter() - started
        return ranked

    def _search_partitions(
        self,
        partitions: list[str],
        query_vector_by_generation: dict[str, list[float]],
        top_k: int,
        visited: set[str],
        profile: QueryProfile | None,
        allow_parallel: bool = True,
    ) -> list[SearchHit]:
        unique_partitions = list(dict.fromkeys(partitions))
        if (
            allow_parallel
            and self._partition_pool is not None
            and len(unique_partitions) > 1
        ):
            task_profiles = [
                QueryProfile() if profile is not None else None
                for _ in unique_partitions
            ]
            futures = [
                self._partition_pool.submit(
                    self._search_partitions,
                    [partition_id],
                    query_vector_by_generation,
                    top_k,
                    set(),
                    task_profile,
                    False,
                )
                for partition_id, task_profile
                in zip(unique_partitions, task_profiles, strict=True)
            ]
            parallel_hits: list[SearchHit] = []
            for future, task_profile in zip(futures, task_profiles, strict=True):
                parallel_hits.extend(future.result())
                if profile is not None and task_profile is not None:
                    profile.add_parallel_work(task_profile)
            return parallel_hits

        hits: list[SearchHit] = []
        for partition_id in unique_partitions:
            if partition_id in visited:
                continue
            visited.add(partition_id)
            started = time.perf_counter() if profile is not None else 0.0
            manifest, _ = self._object_reads.run([
                lambda partition_id=partition_id: (
                    self.partition_store.read_current_manifest(partition_id)
                )
            ])[0]
            if profile is not None:
                profile.manifest_read_seconds += time.perf_counter() - started
            if manifest.state == "split":
                hits.extend(self._search_partitions(
                    manifest.children,
                    query_vector_by_generation,
                    top_k,
                    visited,
                    profile,
                    False,
                ))
                continue
            hits.extend(self._search_active_partition(
                manifest,
                query_vector_by_generation,
                top_k,
                visited,
                profile,
            ))
        return hits

    def _search_active_partition(
        self,
        manifest: PartitionManifest,
        query_vector_by_generation: dict[str, list[float]],
        top_k: int,
        visited: set[str],
        profile: QueryProfile | None,
    ) -> list[SearchHit]:
        attempts = 0
        while True:
            query_vector = query_vector_by_generation.get(manifest.active_generation)
            if query_vector is None:
                return []
            try:
                return self._search_manifest(
                    manifest,
                    query_vector,
                    top_k,
                    profile,
                )
            except ReferencedObjectMissing:
                if attempts >= self.max_missing_retries:
                    raise
                attempts += 1
                started = time.perf_counter() if profile is not None else 0.0
                latest, _ = self._object_reads.run([
                    lambda: self.partition_store.read_current_manifest(
                        manifest.partition_id
                    )
                ])[0]
                if profile is not None:
                    profile.manifest_read_seconds += time.perf_counter() - started
                if latest.state == "split":
                    return self._search_partitions(
                        latest.children,
                        query_vector_by_generation,
                        top_k,
                        visited,
                        profile,
                        False,
                    )
                if latest.version <= manifest.version:
                    if attempts >= self.max_missing_retries:
                        raise
                manifest = latest

    def _candidate_partitions(
        self,
        query_vector_by_generation: dict[str, list[float]],
        metadata_filter: dict[str, Any] | None,
        routing_top_k: int | None,
        profile: QueryProfile | None,
    ) -> list[str]:
        started = time.perf_counter() if profile is not None else 0.0
        view = self._object_reads.run([self.routing_store.effective_view])[0]
        if not view:
            partitions = self._object_reads.run([
                self.partition_store.list_partitions
            ])[0]
            if profile is not None:
                profile.routing_read_seconds += time.perf_counter() - started
            return partitions
        if profile is not None:
            profile.routing_read_seconds += time.perf_counter() - started

        started = time.perf_counter() if profile is not None else 0.0
        entries = [entry for entry in view.values() if entry.state == "active"]
        if metadata_filter:
            entries = [
                entry for entry in entries
                if _metadata_matches(entry.metadata_summary, metadata_filter)
            ]
        partitions: list[str] | None = None
        if routing_top_k is not None:
            scored = []
            for entry in entries:
                query_vector = query_vector_by_generation.get(entry.active_generation)
                if query_vector is None or entry.centroid is None:
                    continue
                scored.append((
                    _routing_score(
                        self.routing_metric,
                        query_vector,
                        entry.centroid,
                    ),
                    entry.partition_id,
                ))
            if scored:
                partitions = [
                    partition_id for _, partition_id
                    in sorted(scored, reverse=True)[:routing_top_k]
                ]
        if partitions is None:
            partitions = sorted(entry.partition_id for entry in entries)
        if profile is not None:
            profile.routing_compute_seconds += time.perf_counter() - started
        return partitions

    def _search_manifest(
        self,
        manifest: PartitionManifest,
        query_vector: list[float],
        top_k: int,
        profile: QueryProfile | None,
    ) -> list[SearchHit]:
        segments, visible_source = self._load_persistent_segments(
            manifest,
            top_k,
            profile,
        )
        if not visible_source:
            return []

        while True:
            for segment in segments:
                if segment.needs_search:
                    started = time.perf_counter() if profile is not None else 0.0
                    segment.hits = segment.loaded_index.search(
                        query_vector,
                        segment.limit,
                    )
                    if profile is not None:
                        profile.ann_search_seconds += time.perf_counter() - started
                    segment.needs_search = False

            started = time.perf_counter() if profile is not None else 0.0
            candidates = []
            for segment in segments:
                for hit in segment.hits:
                    if hit.position < 0 or hit.position >= len(segment.ids):
                        raise ValueError(
                            "ANN index returned an out-of-range row position"
                        )
                    record_id = segment.ids[hit.position]
                    if visible_source.get(record_id) != segment.ordinal:
                        continue
                    candidates.append(SearchHit(
                        id=record_id,
                        score=hit.score,
                        partition_id=manifest.partition_id,
                        generation=manifest.active_generation,
                        metadata=hit.metadata,
                        text=hit.text,
                    ))
            ranked = sorted(
                candidates,
                key=lambda hit: hit.score,
                reverse=True,
            )
            kth_score = (
                ranked[top_k - 1].score
                if len(ranked) >= top_k
                else None
            )
            expandable = [
                segment
                for segment in segments
                if segment.can_expand
                and (
                    kth_score is None
                    or segment.hits[-1].score >= kth_score
                )
            ]
            if not expandable:
                if profile is not None:
                    profile.visibility_merge_seconds += (
                        time.perf_counter() - started
                    )
                return ranked[:top_k]
            for segment in expandable:
                segment.limit = min(
                    segment.count,
                    max(segment.limit + 1, segment.limit * 2),
                )
                segment.needs_search = True
            if profile is not None:
                profile.visibility_merge_seconds += time.perf_counter() - started

    def _load_persistent_segments(
        self,
        manifest: PartitionManifest,
        top_k: int,
        profile: QueryProfile | None,
    ) -> tuple[list["_PersistentSegment"], dict[str, int]]:
        """Load one manifest's objects through the shared bounded read pool."""

        references: list[tuple[int, SegmentRef | DeltaRef, str]] = []
        if manifest.base is not None:
            references.append((0, manifest.base, manifest.base.index_key))
        references.extend(
            (ordinal, delta, delta.mini_index_key)
            for ordinal, delta in enumerate(manifest.visible_deltas, start=1)
        )
        deltas = manifest.visible_deltas

        started = time.perf_counter() if profile is not None else 0.0
        tombstone_lists = self._object_reads.run([
            lambda delta=delta: self.partition_store.read_tombstones(delta)
            for delta in deltas
        ])
        if profile is not None:
            profile.tombstone_read_seconds += time.perf_counter() - started
        tombstones_by_ordinal = {
            ordinal: tombstones
            for ordinal, tombstones in enumerate(tombstone_lists, start=1)
        }

        started = time.perf_counter() if profile is not None else 0.0
        id_lists = self._object_reads.run([
            lambda ref=ref: self.partition_store.read_ids(ref)
            for _, ref, _ in references
        ])
        if profile is not None:
            profile.ids_read_seconds += time.perf_counter() - started

        started = time.perf_counter() if profile is not None else 0.0
        visible_source: dict[str, int] = {}
        loaded_references = []
        for (ordinal, ref, index_key), ids in zip(
            references,
            id_lists,
            strict=True,
        ):
            for tombstone in tombstones_by_ordinal.get(ordinal, []):
                visible_source.pop(tombstone, None)
            for record_id in ids:
                visible_source[record_id] = ordinal
            if ids:
                loaded_references.append((ordinal, ref, index_key, ids))
        if profile is not None:
            profile.visibility_merge_seconds += time.perf_counter() - started

        started = time.perf_counter() if profile is not None else 0.0
        index_blobs = self._object_reads.run([
            lambda index_key=index_key: (
                self.partition_store.read_index_blob(index_key)
            )
            for _, _, index_key, _ in loaded_references
        ])
        if profile is not None:
            profile.index_read_seconds += time.perf_counter() - started

        segments: list[_PersistentSegment] = []
        for (ordinal, _, _, ids), index_blob in zip(
            loaded_references,
            index_blobs,
            strict=True,
        ):
            started = time.perf_counter() if profile is not None else 0.0
            loaded_index = self.index.load(index_blob)
            if profile is not None:
                profile.index_deserialize_seconds += time.perf_counter() - started
                profile.loaded_segment_count += 1
            if loaded_index.count != len(ids):
                raise ValueError("ANN index count does not match segment IDs")
            segments.append(_PersistentSegment(
                ordinal=ordinal,
                count=len(ids),
                ids=ids,
                loaded_index=loaded_index,
                limit=min(len(ids), max(1, top_k * 2)),
            ))
        return segments, visible_source

@dataclass
class _PersistentSegment:
    """One persisted base or delta index participating in local top-k."""

    ordinal: int
    count: int
    ids: list[str]
    loaded_index: LoadedANNIndex
    limit: int
    hits: list[LocalIndexHit] = field(default_factory=list)
    needs_search: bool = True

    @property
    def can_expand(self) -> bool:
        """Return whether another search can reveal additional candidates."""

        return (
            self.limit < self.count
            and len(self.hits) >= self.limit
            and bool(self.hits)
        )


def _metadata_matches(
    metadata_summary: dict[str, dict[str, int]],
    metadata_filter: dict[str, Any],
) -> bool:
    for key, expected in metadata_filter.items():
        if metadata_summary.get(key, {}).get(str(expected), 0) <= 0:
            return False
    return True


def _routing_score(
    metric: str,
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("query and routing centroid must have the same dimension")
    if metric == "l2":
        return -sum(
            (a - b) ** 2
            for a, b in zip(left, right, strict=True)
        )
    return _cosine(left, right)


def _cosine(left: list[float], right: list[float]) -> float:
    denom = math.sqrt(sum(v * v for v in left)) * math.sqrt(sum(v * v for v in right))
    if denom == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denom
