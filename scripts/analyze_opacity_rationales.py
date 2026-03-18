#!/usr/bin/env python3
"""Cluster opacity rationales with embeddings, k-means, and UMAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interviewer.analysis.opacity_rationale_clustering import (
    DEFAULT_CLUSTERS_BY_MECHANISM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LEVELS,
    run_opacity_rationale_analysis_by_mechanism,
)
from interviewer.opacity_columns import FORMS, LEVELS, MECHANISM_KEYS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze opacity rationales with embeddings, k-means, and UMAP.",
    )
    parser.add_argument(
        "results_csv",
        nargs="?",
        type=Path,
        default=Path("outputs/runs/exp0.3/results.csv"),
        help="Path to a collected results.csv export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/opacity_rationale_clustering"),
        help="Directory for analysis artifacts.",
    )
    parser.add_argument(
        "--mechanism",
        dest="mechanisms",
        action="append",
        choices=MECHANISM_KEYS,
        help="Restrict analysis to one or more mechanisms. Each mechanism is clustered separately.",
    )
    parser.add_argument(
        "--level",
        dest="levels",
        action="append",
        choices=sorted(LEVELS),
        help="Restrict analysis to one or more levels. Defaults to all levels, including none.",
    )
    parser.add_argument(
        "--form",
        dest="forms",
        action="append",
        choices=sorted(FORMS),
        help="Restrict analysis to one or more forms.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many rationale rows after filtering.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model to use. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=100,
        help="Sentence-transformers encode batch size.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional sentence-transformers device override, e.g. cpu or mps.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help=(
            "Override the number of k-means clusters for every selected mechanism. "
            "Defaults to mechanism-specific values: "
            + ", ".join(
                f"{mechanism}={count}"
                for mechanism, count in DEFAULT_CLUSTERS_BY_MECHANISM.items()
            )
            + "."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for sampling, k-means, and UMAP.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter. Clio uses 15.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist parameter. Clio uses 0.",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Ignore any cached embeddings in the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        artifacts_by_mechanism = run_opacity_rationale_analysis_by_mechanism(
            args.results_csv,
            output_dir=args.output_dir,
            mechanisms=args.mechanisms,
            levels=args.levels or DEFAULT_LEVELS,
            forms=args.forms,
            sample_size=args.sample_size,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_device=args.embedding_device,
            n_clusters=args.clusters,
            random_state=args.random_state,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            force_reembed=args.force_reembed,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for mechanism, artifacts in artifacts_by_mechanism.items():
        print(f"Mechanism: {mechanism}")
        print(f"Rationale rows: {artifacts.rationale_rows_path}")
        print(f"Embeddings: {artifacts.embeddings_path}")
        print(f"Clustered points: {artifacts.clustered_points_path}")
        print(f"Cluster summary: {artifacts.cluster_summary_path}")
        print(f"UMAP figure: {artifacts.figure_path}")
        print(f"Params: {artifacts.params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
