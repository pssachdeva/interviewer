#!/usr/bin/env python3
"""Label rationale clusters with short phrases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interviewer.analysis.opacity_rationale_clustering import TEXT_SOURCES
from interviewer.clustering.cluster_labeling import ClusterLabelConfig, label_clusters, load_cluster_label_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label rationale clusters with short phrases.")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/clustering/cluster_labels.yaml"),
        help="Path to a clustering label config YAML.",
    )
    parser.add_argument(
        "--text-source",
        choices=TEXT_SOURCES,
        default=None,
        help="Override whether cluster labeling uses representative rationales or evidence.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_cluster_label_config(args.config)
        if args.text_source is not None:
            config = ClusterLabelConfig(
                name=config.name,
                provider=config.provider,
                model=config.model,
                prompt_file=config.prompt_file,
                input_root=config.input_root,
                output_file=config.output_file,
                text_source=args.text_source,
                reasoning_effort=config.reasoning_effort,
                samples_per_cluster=config.samples_per_cluster,
                request=config.request,
                source_path=config.source_path,
            )
        rows = label_clusters(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Config: {config.source_path}")
    print(f"Text source: {config.text_source}")
    print(f"Output file: {config.output_file}")
    print(f"Labeled clusters: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
