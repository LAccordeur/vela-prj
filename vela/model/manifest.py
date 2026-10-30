"""Partition manifest models.

`PartitionManifest` is the authoritative query-visible state for a partition.
Queries must follow the manifest and must not infer visibility from object
existence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SegmentRef:
    """Reference to an immutable base index bundle.

    ``metadata_key`` is omitted for vector-only records whose metadata is empty
    and whose text is ``None``.
    """

    generation: str
    version: str
    vectors_key: str
    ids_key: str
    metadata_key: str | None
    index_key: str
    summary_key: str | None
    count: int


@dataclass(frozen=True)
class DeltaRef:
    """Reference to an immutable committed delta submission bundle.

    Concurrent submissions may share the numeric epoch prefix. The commit id in
    the full epoch label keeps bundles unique, while manifest list order defines
    their committed application order.
    """

    generation: str
    epoch: str
    vectors_key: str
    ids_key: str
    metadata_key: str | None
    tombstones_key: str | None
    mini_index_key: str
    summary_delta_key: str | None
    count: int


@dataclass
class PartitionManifest:
    """Published partition state protected by manifest pointer CAS."""

    partition_id: str
    version: int
    active_generation: str
    latest_base_version: int = 0
    latest_delta_epoch: int = 0
    state: Literal["active", "split"] = "active"
    base: SegmentRef | None = None
    visible_deltas: list[DeltaRef] = field(default_factory=list)
    retained_generations: dict[str, SegmentRef] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    parent_partition_id: str | None = None
    preparation_id: str | None = None


@dataclass(frozen=True)
class ManifestHandle:
    """Pinned manifest version and pointer token observed by an operation."""

    partition_id: str
    manifest_key: str | None
    version: int
    pointer_token: str | None
