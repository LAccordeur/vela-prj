"""GIST benchmark for incremental Vela and offline partitioning baselines.

This entry point reuses the SIFT construction benchmark runner so both
datasets execute identical layout, compaction, routing, and query workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.sift_benchmark import _parser as _construction_parser
from benchmarks.sift_benchmark import run_cli


def _parser() -> argparse.ArgumentParser:
    """Return the construction benchmark CLI configured for GIST1M."""

    return _construction_parser(
        default_dataset_dir=Path("/home/yangguo/Dataset/gist"),
        dataset_label="GIST1M",
        description=__doc__,
    )


def main() -> None:
    """Run the GIST CLI benchmark."""

    run_cli(_parser(), "gist")


if __name__ == "__main__":
    main()
