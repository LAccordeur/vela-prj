"""Minimal end-to-end demo for the Vela prototype."""

from __future__ import annotations

import shutil
from pathlib import Path

from vela.execution.bootstrap import BootstrapExecutor
from vela.execution.ingest import IngestExecutor
from vela.execution.query import QueryExecutor
from vela.index.flat import FlatIndex
from vela.model.record import VectorRecord
from vela.partition.hash_policy import HashPartitionPolicy
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.routing_store import RoutingStore


def main() -> None:
    root = Path(".vela_data/basic_demo")
    if root.exists():
        shutil.rmtree(root)

    index = FlatIndex()
    object_store = LocalObjectStore(root)
    partition_store = PartitionStore(object_store, index, dataset_id="demo")
    routing_store = RoutingStore(object_store, dataset_id="demo")
    bootstrap = BootstrapExecutor(partition_store, routing_store)

    bootstrap.create_partition("p0000", "g0")
    bootstrap.create_partition("p0001", "g0")

    ingest = IngestExecutor(partition_store, routing_store, HashPartitionPolicy())
    query = QueryExecutor(partition_store, routing_store, index)

    ingest.ingest(
        [
            VectorRecord("doc-a", [1.0, 0.0], {"tenant": "a"}, "database incident"),
            VectorRecord("doc-b", [0.0, 1.0], {"tenant": "a"}, "revenue report"),
            VectorRecord("doc-c", [0.8, 0.2], {"tenant": "b"}, "latency runbook"),
        ]
    )

    print("Top results:")
    for hit in query.query({"g0": [1.0, 0.0]}, top_k=3):
        print(f"{hit.id} score={hit.score:.3f} partition={hit.partition_id} text={hit.text}")


if __name__ == "__main__":
    main()
