"""Persistence wrapper for ANN index objects."""

from __future__ import annotations

from vela.index.base import ANNIndex
from vela.model.record import VectorRecord
from vela.storage.object_store import ObjectStore


class ANNIndexStore:
    """Build and read ANN index objects via an `ObjectStore`."""

    def __init__(self, store: ObjectStore, index: ANNIndex) -> None:
        self.store = store
        self.index = index

    def build_and_write(self, key: str, records: list[VectorRecord]) -> None:
        """Build an index for records and write it as an immutable object."""

        self.store.put(key, self.index.build(records), if_absent=True)

    def read_index(self, key: str) -> bytes:
        """Read a serialized index object."""

        return self.store.get(key)
