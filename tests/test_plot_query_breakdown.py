"""Tests for benchmark query-breakdown plotting."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from benchmarks.plot_query_breakdown import (
    extract_fanout_one_breakdown,
    extract_latency_series,
    plot_query_breakdown,
)


class QueryBreakdownPlotTest(unittest.TestCase):
    """Covers phase grouping, fanout extraction, and rendered outputs."""

    def test_extracts_grouped_phases_and_latency_series(self) -> None:
        result = _result_fixture()

        breakdown = extract_fanout_one_breakdown(result)
        _, grouped, total = breakdown["current-uncompacted"]
        self.assertEqual(grouped["Routing"], 3.0)
        self.assertEqual(grouped["Metadata GET"], 8.0)
        self.assertEqual(grouped["Index GET"], 10.0)
        self.assertEqual(grouped["Index deserialize"], 5.0)
        self.assertEqual(grouped["ANN search"], 2.0)
        self.assertEqual(grouped["Merge & other"], 2.0)
        self.assertEqual(total, 30.0)

        fanouts, series = extract_latency_series(result)
        self.assertEqual(fanouts, [1, 4])
        self.assertEqual(
            series["current-uncompacted"],
            ("Current-U", [30.0, 120.0]),
        )

    def test_rejects_parallel_phase_work_for_stacked_latency(self) -> None:
        result = _result_fixture()
        result["scenarios"]["current"]["uncompacted"]["queries"]["1"][
            "phase_accounting"
        ] = "worker-seconds"

        with self.assertRaisesRegex(ValueError, "not wall time"):
            extract_fanout_one_breakdown(result)

    def test_writes_png_and_pdf_figures(self) -> None:
        result = _result_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for suffix in (".png", ".pdf"):
                with self.subTest(suffix=suffix):
                    output = root / f"query-breakdown{suffix}"
                    plot_query_breakdown(result, output, title="Test run")
                    self.assertTrue(output.exists())
                    self.assertGreater(output.stat().st_size, 1_000)


def _result_fixture() -> dict:
    queries = {
        "1": _metrics(30.0, "wall"),
        "4": _metrics(120.0, "worker-seconds"),
    }
    return {
        "scenarios": {
            "current": {
                "uncompacted": {"queries": deepcopy(queries)},
                "compacted": {"queries": deepcopy(queries)},
            },
            "batch_local_semantic": {
                "uncompacted": {"queries": deepcopy(queries)},
                "compacted": {"queries": deepcopy(queries)},
            },
            "global_clustering": {"queries": deepcopy(queries)},
            "block": {"queries": deepcopy(queries)},
        }
    }


def _metrics(mean_ms: float, accounting: str) -> dict:
    return {
        "mean_ms": mean_ms,
        "phase_accounting": accounting,
        "mean_breakdown_ms": {
            "routing_read": 2.0,
            "routing_compute": 1.0,
            "manifest_read": 3.0,
            "tombstone_read": 1.0,
            "ids_read": 4.0,
            "index_read": 10.0,
            "index_deserialize": 5.0,
            "ann_search": 2.0,
            "visibility_merge": 1.0,
            "global_merge": 0.5,
            "other": 0.5,
        },
    }


if __name__ == "__main__":
    unittest.main()
