#!/usr/bin/env python3
"""Collect status and available outputs for a submitted batch experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interviewer.batch.orchestrator import collect_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect an OpenAI batch experiment")
    parser.add_argument("experiment", type=Path, help="Path to experiment YAML")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Collect the test run (first-25-transcripts submission) for this experiment",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        result = collect_experiment(args.experiment, test_mode=args.test)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Experiment: {result.manifest.experiment_name}")
    print(f"Batch ID: {result.manifest.batch_id}")
    print(f"Status: {result.status.status}")
    print(f"Status snapshot: {result.status_path}")

    if result.output_path:
        print(f"Output file: {result.output_path}")
    if result.error_path:
        print(f"Error file: {result.error_path}")
    if result.csv_path:
        print(f"CSV file: {result.csv_path}")

    if not result.is_terminal:
        print("Batch is not in terminal state yet; run this command again later.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
