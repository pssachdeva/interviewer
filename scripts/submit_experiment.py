#!/usr/bin/env python3
"""Submit an experiment config to OpenAI Batch API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interviewer.batch.orchestrator import submit_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit an OpenAI batch experiment")
    parser.add_argument("experiment", type=Path, help="Path to experiment YAML")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Submit only the first 25 transcripts and write artifacts under a test run directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        manifest = submit_experiment(args.experiment, test_mode=args.test)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Submitted experiment: {manifest.experiment_name}")
    print(f"Batch ID: {manifest.batch_id}")
    print(f"Transcript count: {manifest.transcript_count}")
    print(f"Run directory: {manifest.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
