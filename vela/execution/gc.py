"""Conservative garbage collection for Vela-managed object layouts."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from vela.storage.object_store import ObjectStore
from vela.utils.json_codec import from_json_bytes


@dataclass(frozen=True)
class GarbageCollectionReport:
    """Reachability and deletion result for one garbage-collection pass."""

    dataset_id: str
    dry_run: bool
    scanned_keys: list[str]
    reachable_keys: list[str]
    garbage_keys: list[str]
    deferred_keys: list[str]
    deleted_keys: list[str]
    ignored_keys: list[str]
    aborted: bool = False
    reason: str | None = None


class GarbageCollector:
    """Mark reachable Vela objects and sweep old unreachable objects."""

    def __init__(
        self,
        store: ObjectStore,
        dataset_id: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.dataset_id = dataset_id
        self.clock = clock

    def collect(
        self,
        retain_manifest_count: int = 2,
        retain_routing_checkpoint_count: int = 1,
        grace_period_seconds: float = 3600.0,
        dry_run: bool = True,
    ) -> GarbageCollectionReport:
        """Return a dry-run report or delete old objects outside the retained graph."""

        if retain_manifest_count < 1:
            raise ValueError("retain_manifest_count must be at least 1")
        if retain_routing_checkpoint_count < 1:
            raise ValueError("retain_routing_checkpoint_count must be at least 1")
        if grace_period_seconds < 0:
            raise ValueError("grace_period_seconds must not be negative")

        dataset_prefix = f"datasets/{self.dataset_id}"
        all_keys = self.store.list(dataset_prefix)
        managed_keys = {key for key in all_keys if self._is_managed_key(key)}
        ignored_keys = sorted(set(all_keys) - managed_keys)
        pointer_keys = {
            key
            for key in managed_keys
            if key.endswith("/manifest.current")
            or key.endswith("/routing_summary.current")
        }
        pointer_snapshot = {
            key: self.store.read_pointer(key)
            for key in pointer_keys
        }

        reachable = self._partition_reachable_keys(
            managed_keys,
            retain_manifest_count,
            pointer_snapshot,
        )
        reachable.update(self._routing_reachable_keys(
            managed_keys,
            retain_routing_checkpoint_count,
            pointer_snapshot,
        ))

        garbage = sorted(managed_keys - reachable)
        cutoff = self.clock() - grace_period_seconds
        eligible = []
        deferred = []
        for key in garbage:
            if self.store.stat(key).last_modified <= cutoff:
                eligible.append(key)
            else:
                deferred.append(key)

        if dry_run:
            return self._report(
                managed_keys,
                reachable,
                garbage,
                deferred,
                [],
                ignored_keys,
                dry_run=True,
            )

        if not self._pointers_unchanged(pointer_snapshot):
            return self._report(
                managed_keys,
                reachable,
                garbage,
                deferred,
                [],
                ignored_keys,
                dry_run=False,
                aborted=True,
                reason="current pointer changed during garbage-collection scan",
            )

        deleted = []
        for key in sorted(eligible, key=lambda item: item.endswith(".current")):
            self.store.delete(key)
            deleted.append(key)
        return self._report(
            managed_keys,
            reachable,
            garbage,
            deferred,
            deleted,
            ignored_keys,
            dry_run=False,
        )

    def _partition_reachable_keys(
        self,
        managed_keys: set[str],
        retain_manifest_count: int,
        pointer_snapshot: dict,
    ) -> set[str]:
        manifest_pointers = {
            key: value
            for key, value in pointer_snapshot.items()
            if key.endswith("/manifest.current") and value.value is not None
        }
        current_manifests = {}
        partition_pointer_keys = {}
        for pointer_key, pointer in manifest_pointers.items():
            partition_id = pointer_key.split("/")[-2]
            current_manifests[partition_id] = self._read_json(pointer.value)
            partition_pointer_keys[partition_id] = pointer_key

        reachable_partitions = {
            partition_id
            for partition_id, manifest in current_manifests.items()
            if manifest.get("parent_partition_id") is None
        }
        changed = True
        while changed:
            changed = False
            for parent_id in list(reachable_partitions):
                parent = current_manifests[parent_id]
                if parent.get("state", "active") != "split":
                    continue
                for child_id in parent.get("children", []):
                    child = current_manifests.get(child_id)
                    if (
                        child is not None
                        and child.get("parent_partition_id") == parent_id
                        and child_id not in reachable_partitions
                    ):
                        reachable_partitions.add(child_id)
                        changed = True

        manifest_keys_by_partition: dict[str, list[str]] = {}
        manifest_prefix = f"datasets/{self.dataset_id}/partitions/"
        for key in managed_keys:
            if not key.startswith(manifest_prefix) or "/manifests/" not in key:
                continue
            partition_id = key.split("/")[3]
            manifest_keys_by_partition.setdefault(partition_id, []).append(key)

        reachable = set()
        for partition_id in reachable_partitions:
            pointer_key = partition_pointer_keys[partition_id]
            current_key = manifest_pointers[pointer_key].value
            reachable.add(pointer_key)
            selected = [current_key]
            other_keys = [
                key
                for key in manifest_keys_by_partition.get(partition_id, [])
                if key != current_key
            ]
            other_keys.sort(
                key=lambda key: (self._read_json(key).get("version", -1), key),
                reverse=True,
            )
            selected.extend(other_keys[:retain_manifest_count - 1])
            for manifest_key in selected:
                reachable.add(manifest_key)
                reachable.update(self._manifest_object_keys(self._read_json(manifest_key)))
        return reachable

    def _routing_reachable_keys(
        self,
        managed_keys: set[str],
        retain_checkpoint_count: int,
        pointer_snapshot: dict,
    ) -> set[str]:
        reachable = set()
        pointer_key = (
            f"datasets/{self.dataset_id}/routing/routing_summary.current"
        )
        pointer = pointer_snapshot.get(pointer_key)
        current_epoch = 0
        current_key = None
        if pointer is not None and pointer.value is not None:
            reachable.add(pointer_key)
            current_key = pointer.value
            current_epoch = self._read_json(current_key)["epoch"]

        checkpoint_prefix = f"datasets/{self.dataset_id}/routing/checkpoints/"
        checkpoint_keys = [
            key
            for key in managed_keys
            if key.startswith(checkpoint_prefix)
        ]
        selected = []
        if current_key is not None:
            selected.append(current_key)
            other_keys = [key for key in checkpoint_keys if key != current_key]
            other_keys.sort(
                key=lambda key: (self._read_json(key).get("epoch", -1), key),
                reverse=True,
            )
            selected.extend(other_keys[:retain_checkpoint_count - len(selected)])
        reachable.update(selected)

        update_prefix = (
            f"datasets/{self.dataset_id}/routing/partition_updates/"
            f"after_{current_epoch:06d}/"
        )
        reachable.update(
            key
            for key in managed_keys
            if key.startswith(update_prefix)
        )
        return reachable

    def _manifest_object_keys(self, manifest: dict) -> set[str]:
        if manifest.get("state", "active") == "split":
            return set()
        keys = set()
        base = manifest.get("base")
        if base:
            keys.update(self._segment_keys(base))
        for delta in manifest.get("visible_deltas", []):
            keys.update(self._delta_keys(delta))
        for retained in manifest.get("retained_generations", {}).values():
            keys.update(self._segment_keys(retained))
        return keys

    @staticmethod
    def _segment_keys(segment: dict) -> set[str]:
        return {
            key
            for key in (
                segment.get("vectors_key"),
                segment.get("ids_key"),
                segment.get("metadata_key"),
                segment.get("index_key"),
                segment.get("summary_key"),
            )
            if key is not None
        }

    @staticmethod
    def _delta_keys(delta: dict) -> set[str]:
        return {
            key
            for key in (
                delta.get("vectors_key"),
                delta.get("ids_key"),
                delta.get("metadata_key"),
                delta.get("tombstones_key"),
                delta.get("mini_index_key"),
                delta.get("summary_delta_key"),
            )
            if key is not None
        }

    def _pointers_unchanged(self, snapshot: dict) -> bool:
        current_keys = {
            key
            for key in self.store.list(f"datasets/{self.dataset_id}")
            if self._is_managed_key(key)
            and (
                key.endswith("/manifest.current")
                or key.endswith("/routing_summary.current")
            )
        }
        if current_keys != set(snapshot):
            return False
        for key, pointer in snapshot.items():
            current = self.store.read_pointer(key)
            if current.token != pointer.token or current.value != pointer.value:
                return False
        return True

    def _is_managed_key(self, key: str) -> bool:
        prefix = f"datasets/{self.dataset_id}/"
        if not key.startswith(prefix):
            return False
        relative = key[len(prefix):]
        parts = relative.split("/")
        if len(parts) >= 3 and parts[0] == "partitions":
            if len(parts) == 3:
                return parts[2] == "manifest.current"
            if len(parts) == 4 and parts[2] == "manifests":
                return (
                    parts[3].startswith("manifest_")
                    and parts[3].endswith(".json")
                )
            if len(parts) == 6 and parts[2] == "base":
                return parts[5] in {
                    "vectors.npy",
                    "vectors.json",
                    "ids.json",
                    "metadata.json",
                    "index.json",
                    "index.faiss",
                    "summary.json",
                }
            if len(parts) == 6 and parts[2] == "delta":
                return parts[5] in {
                    "vectors.npy",
                    "vectors.json",
                    "ids.json",
                    "metadata.json",
                    "tombstones.json",
                    "mini_index.json",
                    "mini_index.faiss",
                    "summary_delta.json",
                }
            return False
        if parts[:2] == ["routing", "routing_summary.current"]:
            return len(parts) == 2
        if len(parts) == 3 and parts[:2] == ["routing", "checkpoints"]:
            return (
                parts[2].startswith("routing_summary_")
                and parts[2].endswith(".json")
            )
        return (
            len(parts) == 4
            and parts[:2] == ["routing", "partition_updates"]
            and parts[2].startswith("after_")
            and parts[3].startswith("update_")
            and parts[3].endswith(".json")
        )

    def _read_json(self, key: str) -> dict:
        return from_json_bytes(self.store.get(key))

    def _report(
        self,
        managed_keys: set[str],
        reachable: set[str],
        garbage: list[str],
        deferred: list[str],
        deleted: list[str],
        ignored: list[str],
        *,
        dry_run: bool,
        aborted: bool = False,
        reason: str | None = None,
    ) -> GarbageCollectionReport:
        return GarbageCollectionReport(
            dataset_id=self.dataset_id,
            dry_run=dry_run,
            scanned_keys=sorted(managed_keys),
            reachable_keys=sorted(reachable & managed_keys),
            garbage_keys=garbage,
            deferred_keys=deferred,
            deleted_keys=deleted,
            ignored_keys=ignored,
            aborted=aborted,
            reason=reason,
        )
