"""Routing checkpoint and update storage."""

from __future__ import annotations

from dataclasses import asdict

from vela.model.routing import RoutingCheckpoint, RoutingCheckpointHandle, RoutingEntry, RoutingUpdate
from vela.protocol.commit import publish_pointer
from vela.storage.object_store import ObjectStore
from vela.utils.ids import new_commit_id
from vela.utils.json_codec import from_json_bytes, to_json_bytes


class RoutingStore:
    """Stores advisory routing checkpoints and partition-level updates."""

    def __init__(self, store: ObjectStore, dataset_id: str) -> None:
        self.store = store
        self.dataset_id = dataset_id

    def read_current_checkpoint(self) -> tuple[RoutingCheckpoint, RoutingCheckpointHandle]:
        """Read the current routing checkpoint, or an empty checkpoint."""

        pointer = self.store.read_pointer(self._pointer_key())
        if pointer.value is None:
            return RoutingCheckpoint(epoch=0), RoutingCheckpointHandle(None, 0, pointer.token)
        checkpoint = self._read_checkpoint(pointer.value)
        return checkpoint, RoutingCheckpointHandle(pointer.value, checkpoint.epoch, pointer.token)

    def write_partition_update(self, entry: RoutingEntry, commit_id: str | None = None) -> None:
        """Write a partition update and re-emit it if checkpointing races."""

        self._write_update(entry, {}, commit_id)

    def write_split_update(
        self,
        source_entry: RoutingEntry,
        child_entries: dict[str, RoutingEntry],
        commit_id: str | None = None,
    ) -> None:
        """Atomically replace a split source with its active child entries."""

        if source_entry.state != "split":
            raise ValueError("split routing update requires a split source entry")
        if len(child_entries) < 2:
            raise ValueError("split routing update requires at least two child entries")
        for partition_id, entry in child_entries.items():
            if partition_id != entry.partition_id or entry.state != "active":
                raise ValueError("split routing child entries must be active and correctly keyed")
        self._write_update(source_entry, child_entries, commit_id)

    def _write_update(
        self,
        entry: RoutingEntry,
        replacement_entries: dict[str, RoutingEntry],
        commit_id: str | None,
    ) -> None:
        checkpoint, _ = self.read_current_checkpoint()
        self._write_update_for_epoch(
            entry,
            replacement_entries,
            checkpoint.epoch,
            commit_id or new_commit_id(),
        )
        latest, _ = self.read_current_checkpoint()
        if latest.epoch == checkpoint.epoch:
            return
        current = latest.entries.get(entry.partition_id)
        if current is not None and current.manifest_version >= entry.manifest_version:
            return
        self._write_update_for_epoch(
            entry,
            replacement_entries,
            latest.epoch,
            new_commit_id(),
        )

    def _write_update_for_epoch(
        self,
        entry: RoutingEntry,
        replacement_entries: dict[str, RoutingEntry],
        checkpoint_epoch: int,
        commit_id: str,
    ) -> None:
        update = RoutingUpdate(
            checkpoint_epoch=checkpoint_epoch,
            entry=entry,
            replacement_entries=replacement_entries,
        )
        key = (
            f"datasets/{self.dataset_id}/routing/partition_updates/"
            f"after_{checkpoint_epoch:06d}/"
            f"update_cp{checkpoint_epoch:06d}_{entry.partition_id}_"
            f"m{entry.manifest_version:06d}_{commit_id}.json"
        )
        self.store.put(key, to_json_bytes(asdict(update)), if_absent=True)

    def effective_view(self) -> dict[str, RoutingEntry]:
        """Return checkpoint entries with pending updates for its interval applied."""

        checkpoint, _ = self.read_current_checkpoint()
        entries = self._merge_entries(
            checkpoint.entries,
            self.list_pending_updates(checkpoint.epoch),
        )
        return {
            partition_id: entry
            for partition_id, entry in entries.items()
            if entry.state == "active"
        }

    def list_pending_updates(self, checkpoint_epoch: int) -> list[RoutingUpdate]:
        """List updates in the interval extending the supplied checkpoint epoch."""

        prefix = (
            f"datasets/{self.dataset_id}/routing/partition_updates/"
            f"after_{checkpoint_epoch:06d}"
        )
        updates = []
        for key in self.store.list(prefix):
            data = from_json_bytes(self.store.get(key))
            updates.append(
                RoutingUpdate(
                    checkpoint_epoch=data["checkpoint_epoch"],
                    entry=_routing_entry_from_dict(data["entry"]),
                    replacement_entries={
                        partition_id: _routing_entry_from_dict(entry)
                        for partition_id, entry
                        in data.get("replacement_entries", {}).items()
                    },
                )
            )
        return updates

    def build_merged_checkpoint(self) -> RoutingCheckpoint:
        """Build the next checkpoint from the current checkpoint and pending updates."""

        checkpoint, _ = self.read_current_checkpoint()
        entries = self._merge_entries(checkpoint.entries, self.list_pending_updates(checkpoint.epoch))
        return RoutingCheckpoint(epoch=checkpoint.epoch + 1, entries=entries)

    def publish_merged_checkpoint(self, commit_id: str | None = None) -> bool:
        """Merge pending updates and publish the next routing checkpoint with CAS."""

        _, handle = self.read_current_checkpoint()
        checkpoint = self.build_merged_checkpoint()
        return self.publish_checkpoint(handle, checkpoint, commit_id=commit_id)

    def publish_checkpoint(
        self,
        source: RoutingCheckpointHandle,
        checkpoint: RoutingCheckpoint,
        commit_id: str | None = None,
    ) -> bool:
        """Publish a global routing checkpoint with pointer CAS."""

        commit_id = commit_id or new_commit_id()
        key = (
            f"datasets/{self.dataset_id}/routing/checkpoints/"
            f"routing_summary_{checkpoint.epoch:06d}_{commit_id}.json"
        )
        payload = {
            "epoch": checkpoint.epoch,
            "entries": {
                partition_id: asdict(entry)
                for partition_id, entry in checkpoint.entries.items()
            },
        }
        self.store.put(key, to_json_bytes(payload), if_absent=True)
        return publish_pointer(self.store, self._pointer_key(), source.pointer_token, key)

    def _merge_entries(
        self,
        base_entries: dict[str, RoutingEntry],
        updates: list[RoutingUpdate],
    ) -> dict[str, RoutingEntry]:
        entries = dict(base_entries)
        for update in updates:
            current = entries.get(update.entry.partition_id)
            if current is None or update.entry.manifest_version >= current.manifest_version:
                entries[update.entry.partition_id] = update.entry
                for partition_id, replacement in update.replacement_entries.items():
                    existing = entries.get(partition_id)
                    if (
                        existing is None
                        or replacement.manifest_version >= existing.manifest_version
                    ):
                        entries[partition_id] = replacement
        return entries

    def _read_checkpoint(self, key: str) -> RoutingCheckpoint:
        data = from_json_bytes(self.store.get(key))
        return RoutingCheckpoint(
            epoch=data["epoch"],
            entries={
                partition_id: _routing_entry_from_dict(entry)
                for partition_id, entry in data.get("entries", {}).items()
            },
        )

    def _pointer_key(self) -> str:
        return f"datasets/{self.dataset_id}/routing/routing_summary.current"


def _routing_entry_from_dict(data: dict) -> RoutingEntry:
    """Decode routing entries while tolerating the old count-based watermark name."""

    entry = dict(data)
    if (
        "summary_covered_delta_count" in entry
        and "summary_covered_delta_epoch" not in entry
    ):
        entry["summary_covered_delta_epoch"] = entry.pop("summary_covered_delta_count")
    return RoutingEntry(**entry)
