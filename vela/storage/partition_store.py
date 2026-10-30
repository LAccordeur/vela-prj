"""Partition storage and manifest publication protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from vela.index.base import ANNIndex, build_index_profiled
from vela.model.manifest import DeltaRef, ManifestHandle, PartitionManifest, SegmentRef
from vela.model.record import VectorRecord
from vela.protocol.commit import publish_pointer
from vela.protocol.validation import validate_manifest, validate_record_bundle
from vela.storage.object_store import ObjectStore
from vela.storage.vector_codec import (
    decode_float32_vectors,
    encode_float32_vectors,
)
from vela.utils.ids import new_commit_id, version_label
from vela.utils.json_codec import from_json_bytes, to_json_bytes


@dataclass
class BundleWriteProfile:
    """Mutable base-bundle build, serialization, and write timings."""

    index_build_seconds: float = 0.0
    serialize_seconds: float = 0.0
    write_seconds: float = 0.0


class ReferencedObjectMissing(RuntimeError):
    """Raised when a committed manifest references an object that is missing."""

    def __init__(self, key: str) -> None:
        super().__init__(f"manifest referenced missing object: {key}")
        self.key = key


class PartitionStore:
    """Stores partition objects and publishes manifests with pointer CAS."""

    def __init__(self, store: ObjectStore, index: ANNIndex, dataset_id: str) -> None:
        self.store = store
        self.index = index
        self.dataset_id = dataset_id
        extension = index.file_extension
        if not extension.startswith(".") or "/" in extension:
            raise ValueError("ANN index file extension must start with a dot")
        self.index_file_extension = extension

    def create_partition(
        self,
        partition_id: str,
        generation: str,
        base_records: list[VectorRecord] | None = None,
        parent_partition_id: str | None = None,
        preparation_id: str | None = None,
    ) -> PartitionManifest:
        """Create a root partition or a child prepared for a parent split."""

        existing = self.store.read_pointer(self._pointer_key(partition_id))
        if existing.value is not None:
            manifest, _ = self.read_current_manifest(partition_id)
            if (
                manifest.active_generation != generation
                or manifest.parent_partition_id != parent_partition_id
                or manifest.preparation_id != preparation_id
            ):
                raise ValueError(f"partition already exists with incompatible state: {partition_id}")
            return manifest
        commit_id = new_commit_id()
        base = self.write_base_bundle(
            partition_id,
            generation,
            base_records or [],
            commit_id=commit_id,
            version=version_label("b", 1, commit_id),
        )
        manifest = PartitionManifest(
            partition_id=partition_id,
            version=1,
            active_generation=generation,
            latest_base_version=1,
            latest_delta_epoch=0,
            base=base,
            parent_partition_id=parent_partition_id,
            preparation_id=preparation_id,
        )
        manifest_key = self._manifest_key(partition_id, manifest.version, commit_id)
        self._write_manifest(manifest_key, manifest)
        if not publish_pointer(self.store, self._pointer_key(partition_id), None, manifest_key):
            winner, _ = self.read_current_manifest(partition_id)
            if (
                winner.active_generation != generation
                or winner.parent_partition_id != parent_partition_id
                or winner.preparation_id != preparation_id
            ):
                raise ValueError(
                    f"partition already exists with incompatible state: {partition_id}"
                )
            return winner
        return manifest

    def prepare_split_child(
        self,
        partition_id: str,
        generation: str,
        records: list[VectorRecord],
        parent_partition_id: str,
        preparation_id: str,
    ) -> PartitionManifest:
        """Create or rebuild a child owned by one uncommitted split attempt."""

        if not self.partition_exists(partition_id):
            return self.create_partition(
                partition_id,
                generation,
                records,
                parent_partition_id=parent_partition_id,
                preparation_id=preparation_id,
            )

        current, handle = self.read_current_manifest(partition_id)
        if (
            current.parent_partition_id != parent_partition_id
            or current.preparation_id != preparation_id
        ):
            raise ValueError(
                f"child partition already exists for another split: {partition_id}"
            )
        commit_id = new_commit_id()
        next_base_version = current.latest_base_version + 1
        base = self.write_base_bundle(
            partition_id,
            generation,
            records,
            commit_id=commit_id,
            version=version_label("b", next_base_version, commit_id),
        )
        candidate = PartitionManifest(
            partition_id=partition_id,
            version=current.version + 1,
            active_generation=generation,
            latest_base_version=next_base_version,
            latest_delta_epoch=current.latest_delta_epoch,
            base=base,
            parent_partition_id=parent_partition_id,
            preparation_id=preparation_id,
        )
        if not self.publish_manifest(handle, candidate, commit_id=commit_id):
            raise RuntimeError(f"prepared child changed concurrently: {partition_id}")
        return candidate

    def list_partitions(self) -> list[str]:
        """Return active leaf partitions reachable from committed split parents."""

        prefix = f"datasets/{self.dataset_id}/partitions"
        keys = self.store.list(prefix)
        partition_ids = sorted({
            key.split("/")[-2]
            for key in keys
            if key.endswith("manifest.current")
        })
        manifests = {
            partition_id: self.read_current_manifest(partition_id)[0]
            for partition_id in partition_ids
        }
        leaves = []
        for partition_id, manifest in manifests.items():
            if manifest.state != "active":
                continue
            if manifest.parent_partition_id is None:
                leaves.append(partition_id)
                continue
            parent = manifests.get(manifest.parent_partition_id)
            if (
                parent is not None
                and parent.state == "split"
                and partition_id in parent.children
            ):
                leaves.append(partition_id)
        return leaves

    def partition_exists(self, partition_id: str) -> bool:
        """Return whether a partition manifest pointer has been published."""

        return self.store.read_pointer(self._pointer_key(partition_id)).value is not None

    def read_current_manifest(self, partition_id: str) -> tuple[PartitionManifest, ManifestHandle]:
        """Read and pin the current partition manifest."""

        pointer = self.store.read_pointer(self._pointer_key(partition_id))
        if pointer.value is None:
            raise KeyError(f"partition does not exist: {partition_id}")
        manifest = self._read_manifest(pointer.value)
        return manifest, ManifestHandle(
            partition_id=partition_id,
            manifest_key=pointer.value,
            version=manifest.version,
            pointer_token=pointer.token,
        )

    def write_delta_bundle(
        self,
        partition_id: str,
        generation: str,
        records: list[VectorRecord],
        tombstones: list[str],
        commit_id: str | None = None,
        epoch: str | None = None,
    ) -> DeltaRef:
        """Write a candidate delta bundle under fresh immutable keys."""

        validate_record_bundle(records)
        commit_id = commit_id or new_commit_id()
        epoch = epoch or version_label("e", 1, commit_id)
        prefix = self._delta_prefix(partition_id, generation, epoch)
        vectors_key = f"{prefix}/vectors.npy"
        ids_key = f"{prefix}/ids.json"
        metadata_key = _metadata_key(prefix, records)
        tombstones_key = (
            f"{prefix}/tombstones.json"
            if tombstones
            else None
        )
        mini_index_key = f"{prefix}/mini_index{self.index_file_extension}"
        summary_delta_key = f"{prefix}/summary_delta.json"
        self._write_record_objects(vectors_key, ids_key, metadata_key, records)
        if tombstones_key is not None:
            self.store.put(tombstones_key, to_json_bytes(tombstones), if_absent=True)
        self.store.put(mini_index_key, self.index.build(records), if_absent=True)
        self.store.put(summary_delta_key, to_json_bytes(_summary(records)), if_absent=True)
        return DeltaRef(
            generation=generation,
            epoch=epoch,
            vectors_key=vectors_key,
            ids_key=ids_key,
            metadata_key=metadata_key,
            tombstones_key=tombstones_key,
            mini_index_key=mini_index_key,
            summary_delta_key=summary_delta_key,
            count=len(records),
        )

    def write_base_bundle(
        self,
        partition_id: str,
        generation: str,
        records: list[VectorRecord],
        commit_id: str | None = None,
        version: str | None = None,
        profile: BundleWriteProfile | None = None,
    ) -> SegmentRef:
        """Write a candidate base bundle under fresh immutable keys."""

        validate_record_bundle(records)
        commit_id = commit_id or new_commit_id()
        version = version or version_label("b", 1, commit_id)
        prefix = self._base_prefix(partition_id, generation, version)
        vectors_key = f"{prefix}/vectors.npy"
        ids_key = f"{prefix}/ids.json"
        metadata_key = _metadata_key(prefix, records)
        index_key = f"{prefix}/index{self.index_file_extension}"
        summary_key = f"{prefix}/summary.json"
        if profile is None:
            self._write_record_objects(vectors_key, ids_key, metadata_key, records)
            self.store.put(index_key, self.index.build(records), if_absent=True)
            self.store.put(summary_key, to_json_bytes(_summary(records)), if_absent=True)
        else:
            self._write_base_objects_profiled(
                vectors_key,
                ids_key,
                metadata_key,
                index_key,
                summary_key,
                records,
                profile,
            )
        return SegmentRef(
            generation=generation,
            version=version,
            vectors_key=vectors_key,
            ids_key=ids_key,
            metadata_key=metadata_key,
            index_key=index_key,
            summary_key=summary_key,
            count=len(records),
        )


    def read_summary(self, summary_key: str | None) -> dict:
        """Read a partition-local summary object."""

        if summary_key is None:
            return {"count": 0, "centroid": None, "metadata_summary": {}}
        summary = from_json_bytes(self.store.get(summary_key))
        summary.setdefault("metadata_summary", {})
        return summary

    def read_base_summary(self, manifest: PartitionManifest) -> dict:
        """Read the active base summary referenced by a manifest."""

        return self.read_summary(manifest.base.summary_key if manifest.base else None)

    def read_index_blob(self, key: str) -> bytes:
        """Read one manifest-referenced serialized ANN index object."""

        return self._get_referenced_object(key)

    def read_ids(self, ref: SegmentRef | DeltaRef) -> list[str]:
        """Read logical record ids for one manifest-referenced segment."""

        return from_json_bytes(self._get_referenced_object(ref.ids_key))

    def read_tombstones(self, delta: DeltaRef) -> list[str]:
        """Read a delta's tombstones, treating an omitted object as empty."""

        if delta.tombstones_key is None:
            return []
        return from_json_bytes(
            self._get_referenced_object(delta.tombstones_key)
        )

    def publish_manifest(
        self,
        source: ManifestHandle,
        candidate: PartitionManifest,
        commit_id: str | None = None,
    ) -> bool:
        """Write and conditionally publish a candidate manifest."""

        manifest_key = self.write_manifest_candidate(candidate, commit_id)
        return publish_pointer(
            self.store,
            self._pointer_key(candidate.partition_id),
            source.pointer_token,
            manifest_key,
        )

    def write_manifest_candidate(
        self, candidate: PartitionManifest, commit_id: str | None = None,
    ) -> str:
        """Write a validated immutable manifest without changing its current pointer."""

        validate_manifest(candidate)
        key = self._manifest_key(candidate.partition_id, candidate.version, commit_id or new_commit_id())
        self._write_manifest(key, candidate)
        return key

    def materialize_live_records(self, manifest: PartitionManifest) -> list[VectorRecord]:
        """Materialize base plus visible deltas while applying tombstones."""

        records: dict[str, VectorRecord] = {}
        for record in self._read_records(manifest.base):
            records[record.id] = record
        for delta in manifest.visible_deltas:
            for tombstone in self.read_tombstones(delta):
                records.pop(tombstone, None)
            for record in self._read_records(delta):
                records[record.id] = record
        return list(records.values())

    def _write_base_objects_profiled(
        self,
        vectors_key: str,
        ids_key: str,
        metadata_key: str | None,
        index_key: str,
        summary_key: str,
        records: list[VectorRecord],
        profile: BundleWriteProfile,
    ) -> None:
        """Build immutable base payloads while separating CPU and I/O time."""

        serialize_started = time.perf_counter()
        vectors_blob = encode_float32_vectors([record.vector for record in records])
        ids_blob = to_json_bytes([record.id for record in records])
        metadata_blob = (
            to_json_bytes({
                record.id: {"metadata": record.metadata, "text": record.text}
                for record in records
            })
            if metadata_key is not None
            else None
        )
        summary_blob = to_json_bytes(_summary(records))
        profile.serialize_seconds += time.perf_counter() - serialize_started

        artifact = build_index_profiled(self.index, records)
        profile.index_build_seconds += artifact.build_seconds
        profile.serialize_seconds += artifact.serialize_seconds

        write_started = time.perf_counter()
        self.store.put(vectors_key, vectors_blob, if_absent=True)
        self.store.put(ids_key, ids_blob, if_absent=True)
        if metadata_blob is not None:
            self.store.put(metadata_key, metadata_blob, if_absent=True)
        self.store.put(index_key, artifact.blob, if_absent=True)
        self.store.put(summary_key, summary_blob, if_absent=True)
        profile.write_seconds += time.perf_counter() - write_started

    def _write_record_objects(
        self,
        vectors_key: str,
        ids_key: str,
        metadata_key: str | None,
        records: list[VectorRecord],
    ) -> None:
        self.store.put(
            vectors_key,
            encode_float32_vectors([record.vector for record in records]),
            if_absent=True,
        )
        self.store.put(ids_key, to_json_bytes([r.id for r in records]), if_absent=True)
        if metadata_key is not None:
            self.store.put(
                metadata_key,
                to_json_bytes({
                    r.id: {"metadata": r.metadata, "text": r.text}
                    for r in records
                }),
                if_absent=True,
            )

    def _read_records(self, ref: SegmentRef | DeltaRef | None) -> list[VectorRecord]:
        if ref is None:
            return []
        vectors_blob = self._get_referenced_object(ref.vectors_key)
        if ref.vectors_key.endswith(".npy"):
            vectors = decode_float32_vectors(vectors_blob).tolist()
        elif ref.vectors_key.endswith(".json"):
            vectors = from_json_bytes(vectors_blob)
        else:
            raise ValueError(
                f"unsupported vector object format: {ref.vectors_key}"
            )
        ids = from_json_bytes(self._get_referenced_object(ref.ids_key))
        metadata = (
            from_json_bytes(self._get_referenced_object(ref.metadata_key))
            if ref.metadata_key is not None
            else {}
        )
        return [
            VectorRecord(
                id=record_id,
                vector=vector,
                metadata=metadata.get(record_id, {}).get("metadata", {}),
                text=metadata.get(record_id, {}).get("text"),
            )
            for record_id, vector in zip(ids, vectors)
        ]

    def _get_referenced_object(self, key: str) -> bytes:
        try:
            return self.store.get(key)
        except (FileNotFoundError, KeyError) as error:
            raise ReferencedObjectMissing(key) from error

    def _write_manifest(self, key: str, manifest: PartitionManifest) -> None:
        self.store.put(key, to_json_bytes(asdict(manifest)), if_absent=True)

    def _read_manifest(self, key: str) -> PartitionManifest:
        data = from_json_bytes(self.store.get(key))
        base = SegmentRef(**data["base"]) if data.get("base") else None
        deltas = [DeltaRef(**item) for item in data.get("visible_deltas", [])]
        retained = {
            generation: SegmentRef(**ref)
            for generation, ref in data.get("retained_generations", {}).items()
        }
        return PartitionManifest(
            partition_id=data["partition_id"],
            version=data["version"],
            active_generation=data["active_generation"],
            latest_base_version=data.get("latest_base_version", 1),
            latest_delta_epoch=data.get("latest_delta_epoch", len(deltas)),
            state=data.get("state", "active"),
            base=base,
            visible_deltas=deltas,
            retained_generations=retained,
            children=data.get("children", []),
            parent_partition_id=data.get("parent_partition_id"),
            preparation_id=data.get("preparation_id"),
        )

    def _pointer_key(self, partition_id: str) -> str:
        return f"datasets/{self.dataset_id}/partitions/{partition_id}/manifest.current"

    def _manifest_key(self, partition_id: str, version: int, commit_id: str) -> str:
        return (
            f"datasets/{self.dataset_id}/partitions/{partition_id}/manifests/"
            f"manifest_{version:06d}_{commit_id}.json"
        )

    def _base_prefix(self, partition_id: str, generation: str, version: str) -> str:
        return f"datasets/{self.dataset_id}/partitions/{partition_id}/base/{generation}/{version}"

    def _delta_prefix(self, partition_id: str, generation: str, epoch: str) -> str:
        return f"datasets/{self.dataset_id}/partitions/{partition_id}/delta/{generation}/{epoch}"


def _metadata_key(prefix: str, records: list[VectorRecord]) -> str | None:
    """Return an object key only when a bundle carries metadata or text."""

    if any(record.metadata or record.text is not None for record in records):
        return f"{prefix}/metadata.json"
    return None

def _summary(records: list[VectorRecord]) -> dict:
    if not records:
        return {"count": 0, "centroid": None, "metadata_summary": {}}
    matrix = np.asarray(
        [record.vector for record in records],
        dtype=np.float32,
    )
    return {
        "count": len(records),
        "centroid": matrix.mean(axis=0, dtype=np.float64).tolist(),
        "metadata_summary": _metadata_summary(records),
    }


def _metadata_summary(records: list[VectorRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        for key, value in record.metadata.items():
            if _is_summary_value(value):
                bucket = summary.setdefault(key, {})
                label = str(value)
                bucket[label] = bucket.get(label, 0) + 1
    return summary


def _is_summary_value(value) -> bool:
    return isinstance(value, str | int | float | bool) or value is None
