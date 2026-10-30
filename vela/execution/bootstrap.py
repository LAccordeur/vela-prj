"""Execution-layer helpers for dataset and partition bootstrap."""

from __future__ import annotations

from vela.execution.routing import routing_entry_from_summary
from vela.model.manifest import PartitionManifest
from vela.model.record import VectorRecord
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore
from vela.utils.ids import new_commit_id


class BootstrapExecutor:
    """Creates partitions and publishes their initial routing metadata."""

    def __init__(
        self,
        partition_store: PartitionStore,
        routing_store: RoutingStore,
    ) -> None:
        self.partition_store = partition_store
        self.routing_store = routing_store

    def create_partition(
        self,
        partition_id: str,
        generation: str,
        base_records: list[VectorRecord] | None = None,
    ) -> PartitionManifest:
        """Create a root partition and write an initial routing update."""

        manifest = self.partition_store.create_partition(
            partition_id,
            generation,
            base_records=base_records,
        )
        summary = self.partition_store.read_base_summary(manifest)
        self.routing_store.write_partition_update(
            routing_entry_from_summary(manifest, summary),
            commit_id=new_commit_id(),
        )
        return manifest
