"""Plot query phase breakdown and fanout latency from benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = (
    (
        "current-uncompacted",
        "Current-U",
        ("current", "uncompacted", "queries"),
    ),
    (
        "current-compacted",
        "Current-C",
        ("current", "compacted", "queries"),
    ),
    (
        "batch-local-uncompacted",
        "Batch-U",
        ("batch_local_semantic", "uncompacted", "queries"),
    ),
    (
        "batch-local-compacted",
        "Batch-C",
        ("batch_local_semantic", "compacted", "queries"),
    ),
    (
        "global-clustering",
        "Global",
        ("global_clustering", "queries"),
    ),
    (
        "block",
        "Block",
        ("block", "queries"),
    ),
)

PHASE_GROUPS = (
    (
        "Routing",
        ("routing_read", "routing_compute"),
        "#4477AA",
    ),
    (
        "Metadata GET",
        ("manifest_read", "tombstone_read", "ids_read"),
        "#66CCEE",
    ),
    (
        "Index GET",
        ("index_read",),
        "#CC6677",
    ),
    (
        "Index deserialize",
        ("index_deserialize",),
        "#AA3377",
    ),
    (
        "ANN search",
        ("ann_search",),
        "#228833",
    ),
    (
        "Merge & other",
        ("visibility_merge", "global_merge", "other"),
        "#BBBBBB",
    ),
)

LINE_COLORS = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
)


def scenario_measurements(
    result: dict[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return scenario labels and query measurements in display order."""

    scenarios = result.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("benchmark JSON is missing scenarios")

    measurements: dict[str, tuple[str, dict[str, Any]]] = {}
    for scenario_name, label, path in SCENARIOS:
        value: Any = scenarios
        try:
            for component in path:
                value = value[component]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"benchmark JSON is missing queries for {scenario_name}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"invalid query measurements for {scenario_name}")
        measurements[scenario_name] = (label, value)
    return measurements


def extract_fanout_one_breakdown(
    result: dict[str, Any],
) -> dict[str, tuple[str, dict[str, float], float]]:
    """Group fanout-one wall-clock phases for a stacked latency chart."""

    breakdown: dict[str, tuple[str, dict[str, float], float]] = {}
    for scenario_name, (label, measurements) in scenario_measurements(
        result
    ).items():
        try:
            metrics = measurements["1"]
            phases = metrics["mean_breakdown_ms"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"fanout=1 phase breakdown missing for {scenario_name}"
            ) from error
        accounting = metrics.get("phase_accounting", "wall")
        if accounting != "wall":
            raise ValueError(
                f"fanout=1 phases for {scenario_name} use {accounting}, not wall time"
            )
        grouped = {
            group_name: sum(float(phases[phase]) for phase in phase_names)
            for group_name, phase_names, _ in PHASE_GROUPS
        }
        breakdown[scenario_name] = (
            label,
            grouped,
            float(metrics["mean_ms"]),
        )
    return breakdown


def extract_latency_series(
    result: dict[str, Any],
) -> tuple[list[int], dict[str, tuple[str, list[float]]]]:
    """Return common numeric fanouts and mean wall-clock latency series."""

    measurements = scenario_measurements(result)
    fanout_sets = [
        {
            int(fanout)
            for fanout in values
            if str(fanout).isdigit()
        }
        for _, values in measurements.values()
    ]
    common_fanouts = sorted(set.intersection(*fanout_sets))
    if not common_fanouts:
        raise ValueError("benchmark JSON has no common numeric fanouts")

    series: dict[str, tuple[str, list[float]]] = {}
    for scenario_name, (label, values) in measurements.items():
        series[scenario_name] = (
            label,
            [float(values[str(fanout)]["mean_ms"]) for fanout in common_fanouts],
        )
    return common_fanouts, series


def plot_query_breakdown(
    result: dict[str, Any],
    output: Path,
    *,
    title: str | None = None,
    dpi: int = 200,
) -> None:
    """Write a two-panel query breakdown and fanout-latency figure."""

    if output.suffix.lower() not in {".pdf", ".png", ".svg"}:
        raise ValueError("output extension must be .pdf, .png, or .svg")
    if dpi < 1:
        raise ValueError("dpi must be positive")

    breakdown = extract_fanout_one_breakdown(result)
    fanouts, latency_series = extract_latency_series(result)
    figure, (breakdown_axis, latency_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.2),
        gridspec_kw={"width_ratios": (1.2, 1.0)},
    )

    scenario_names = list(breakdown)
    y_positions = np.arange(len(scenario_names))
    left = np.zeros(len(scenario_names), dtype=np.float64)
    for group_name, _, color in PHASE_GROUPS:
        values = np.asarray(
            [breakdown[name][1][group_name] for name in scenario_names],
            dtype=np.float64,
        )
        breakdown_axis.barh(
            y_positions,
            values,
            left=left,
            height=0.68,
            label=group_name,
            color=color,
            edgecolor="white",
            linewidth=0.35,
        )
        left += values

    totals = [breakdown[name][2] for name in scenario_names]
    label_offset = max(totals) * 0.018
    for position, total in zip(y_positions, totals, strict=True):
        breakdown_axis.text(
            total + label_offset,
            position,
            f"{total:.0f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    breakdown_axis.set_yticks(
        y_positions,
        [breakdown[name][0] for name in scenario_names],
    )
    breakdown_axis.invert_yaxis()
    breakdown_axis.set_xlim(0.0, max(totals) * 1.16)
    breakdown_axis.set_xlabel("Mean query latency (ms)")
    breakdown_axis.set_title("(a) Fanout=1 latency breakdown")
    breakdown_axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    breakdown_axis.set_axisbelow(True)
    breakdown_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    for color, (scenario_name, (label, values)) in zip(
        LINE_COLORS,
        latency_series.items(),
        strict=True,
    ):
        latency_axis.plot(
            fanouts,
            values,
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=color,
            label=label,
        )
    latency_axis.set_xticks(fanouts)
    latency_axis.set_xlabel("Probed partitions (fanout)")
    latency_axis.set_ylabel("Mean query latency (ms)")
    latency_axis.set_title("(b) End-to-end latency vs. fanout")
    latency_axis.grid(color="#DDDDDD", linewidth=0.7)
    latency_axis.set_axisbelow(True)
    latency_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    if title:
        figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.0, 0.1, 1.0, 0.96 if title else 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    """Load one benchmark result and write its query figure."""

    args = _parser().parse_args()
    result = json.loads(args.results_json.read_text(encoding="utf-8"))
    plot_query_breakdown(
        result,
        args.output,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
