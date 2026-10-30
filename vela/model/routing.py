"""Routing metadata models.

Routing metadata is advisory. It helps choose candidate partitions, while
`PartitionManifest` remains the correctness boundary for query visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RoutingEntry:
    """Compact routing summary for one partition state."""

    partition_id: str
    manifest_version: int
    active_generation: str
    count: int
    manifest_key: str | None = None
    summary_covered_delta_epoch: int = 0
    state: Literal["active", "split"] = "active"
    centroid: list[float] | None = None
    metadata_summary: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class RoutingCheckpoint:
    """Checkpointed global routing view."""

    epoch: int
    entries: dict[str, RoutingEntry] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingCheckpointHandle:
    """Pinned routing checkpoint pointer state."""

    checkpoint_key: str | None
    epoch: int
    pointer_token: str | None


@dataclass(frozen=True)
class RoutingUpdate:
    """Replacement or structural routing update published after a manifest commit."""

    checkpoint_epoch: int
    entry: RoutingEntry
    replacement_entries: dict[str, RoutingEntry] = field(default_factory=dict)
