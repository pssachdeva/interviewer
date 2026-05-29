"""CLio-style clustering utilities for opacity rationales."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px

from interviewer.opacity_columns import (
    FORMS,
    LEVELS,
    MECHANISM_KEYS,
    evidence_column,
    form_column,
    level_column,
    rationale_column,
)

DEFAULT_LEVELS = ("potential", "clear")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_N_CLUSTERS = 10
TEXT_SOURCES = ("rationale", "evidence")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "almost",
    "also",
    "although",
    "always",
    "among",
    "around",
    "because",
    "before",
    "being",
    "between",
    "brief",
    "could",
    "describe",
    "does",
    "doesnt",
    "doing",
    "dont",
    "evidence",
    "explicitly",
    "from",
    "however",
    "interactions",
    "interpersonal",
    "into",
    "just",
    "lack",
    "less",
    "more",
    "mostly",
    "much",
    "other",
    "others",
    "people",
    "practice",
    "practices",
    "rather",
    "relevant",
    "says",
    "show",
    "shows",
    "speaker",
    "still",
    "such",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "transcript",
    "use",
    "used",
    "uses",
    "using",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
    "work",
    "worker",
    "workers",
    "would",
}


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Paths written by one rationale-clustering run."""

    rationale_rows_path: Path
    embeddings_path: Path
    clustered_points_path: Path
    cluster_summary_path: Path
    figure_path: Path
    params_path: Path


@dataclass(frozen=True)
class TaskActivityAnalysisArtifacts:
    """Paths written by one task-activity clustering run."""

    activity_rows_path: Path
    embeddings_path: Path
    clustered_points_path: Path
    cluster_summary_path: Path
    figure_path: Path
    params_path: Path


def load_results_csv(path: str | Path) -> pd.DataFrame:
    """Load an exported opacity results CSV."""
    return pd.read_csv(path)


def mechanism_label(mechanism: str) -> str:
    """Convert a mechanism key into a human-readable label."""
    return mechanism.replace("_opacity", "").replace("_", " ").title()


def normalize_text(value: str) -> str:
    """Collapse whitespace for more stable embedding/cache keys."""
    return " ".join(str(value).split())


def normalize_evidence(value: str) -> str:
    """Normalize exported evidence strings for embedding."""
    parts = [part.strip() for part in str(value).split("||")]
    parts = [part for part in parts if part]
    return " [SEP] ".join(parts)


def parse_task_activities(value: object) -> list[str]:
    """Parse a task-activities cell into a normalized list of strings."""
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = text

    if isinstance(payload, str):
        normalized = normalize_text(payload)
        return [normalized] if normalized else []

    if not isinstance(payload, list):
        raise ValueError("`task_activities` values must be JSON arrays or strings.")

    activities: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError("`task_activities` arrays must contain only strings.")
        normalized = normalize_text(item)
        if normalized:
            activities.append(normalized)
    return activities


def expand_opacity_rationales(
    results: pd.DataFrame,
    mechanisms: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    forms: Sequence[str] | None = None,
    text_source: str = "rationale",
) -> pd.DataFrame:
    """Reshape wide per-mechanism rationale columns into one long table."""
    if text_source not in TEXT_SOURCES:
        allowed = ", ".join(TEXT_SOURCES)
        raise ValueError(f"`text_source` must be one of: {allowed}")

    selected_mechanisms = tuple(mechanisms or MECHANISM_KEYS)
    selected_levels = tuple(levels or DEFAULT_LEVELS)
    invalid_levels = sorted(set(selected_levels) - LEVELS)
    if invalid_levels:
        raise ValueError(f"Unsupported levels: {invalid_levels}")

    selected_forms = tuple(forms) if forms is not None else None
    if selected_forms is not None:
        invalid_forms = sorted(set(selected_forms) - FORMS)
        if invalid_forms:
            raise ValueError(f"Unsupported forms: {invalid_forms}")

    base_columns = [
        column
        for column in ("transcript_id", "transcript", "summary", "model")
        if column in results.columns
    ]

    frames: list[pd.DataFrame] = []
    for mechanism in selected_mechanisms:
        required = [
            level_column(mechanism),
            form_column(mechanism),
            rationale_column(mechanism),
            evidence_column(mechanism),
        ]
        missing = [column for column in required if column not in results.columns]
        if missing:
            raise ValueError(
                f"Results CSV is missing columns for {mechanism}: {missing}"
            )

        mechanism_df = results[base_columns].copy()
        mechanism_df["mechanism"] = mechanism
        mechanism_df["mechanism_label"] = mechanism_label(mechanism)
        mechanism_df["level"] = results[level_column(mechanism)].fillna("")
        mechanism_df["form"] = results[form_column(mechanism)].fillna("")
        mechanism_df["rationale"] = (
            results[rationale_column(mechanism)].fillna("").map(normalize_text)
        )
        mechanism_df["evidence"] = (
            results[evidence_column(mechanism)].fillna("").map(normalize_evidence)
        )
        mechanism_df["row_id"] = (
            mechanism_df["transcript_id"].astype(str) + ":" + mechanism
        )
        frames.append(mechanism_df)

    long_df = pd.concat(frames, ignore_index=True)
    mask = long_df[text_source].ne("") & long_df["level"].isin(selected_levels)
    if selected_forms is not None:
        mask &= long_df["form"].isin(selected_forms)

    filtered = long_df.loc[mask].copy()
    filtered.reset_index(drop=True, inplace=True)
    filtered["embedding_input"] = filtered[text_source]
    filtered["embedding_text_source"] = text_source
    return filtered


def expand_task_activity_results(
    results: pd.DataFrame,
    *,
    mechanisms: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    forms: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Explode task-activity results into one row per activity string."""
    required = {"transcript_id", "mechanism", "level", "form", "task_activities"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Task-activity results CSV is missing columns: {missing}")

    selected_mechanisms = tuple(mechanisms or MECHANISM_KEYS)
    invalid_mechanisms = sorted(set(selected_mechanisms) - set(MECHANISM_KEYS))
    if invalid_mechanisms:
        raise ValueError(f"Unsupported mechanisms: {invalid_mechanisms}")

    selected_levels = tuple(levels or DEFAULT_LEVELS)
    invalid_levels = sorted(set(selected_levels) - LEVELS)
    if invalid_levels:
        raise ValueError(f"Unsupported levels: {invalid_levels}")

    selected_forms = tuple(forms) if forms is not None else None
    if selected_forms is not None:
        invalid_forms = sorted(set(selected_forms) - FORMS)
        if invalid_forms:
            raise ValueError(f"Unsupported forms: {invalid_forms}")

    base_columns = [
        column
        for column in ("transcript_id", "transcript", "mechanism", "level", "form")
        if column in results.columns
    ]

    filtered = results.loc[
        results["mechanism"].isin(selected_mechanisms)
        & results["level"].isin(selected_levels)
    ].copy()
    if selected_forms is not None:
        filtered = filtered.loc[filtered["form"].isin(selected_forms)].copy()

    rows: list[dict[str, object]] = []
    for _, row in filtered.iterrows():
        activities = parse_task_activities(row["task_activities"])
        for activity_index, activity in enumerate(activities):
            exploded: dict[str, object] = {
                column: row[column] for column in base_columns
            }
            mechanism = str(row["mechanism"])
            transcript_id = str(row["transcript_id"])
            exploded["mechanism_label"] = mechanism_label(mechanism)
            exploded["activity_index"] = activity_index
            exploded["task_activity"] = activity
            exploded["row_id"] = f"{transcript_id}:{mechanism}:{activity_index}"
            exploded["embedding_input"] = activity
            exploded["embedding_text_source"] = "task_activity"
            rows.append(exploded)

    expected_columns = base_columns + [
        "mechanism_label",
        "activity_index",
        "task_activity",
        "row_id",
        "embedding_input",
        "embedding_text_source",
    ]
    return pd.DataFrame(rows, columns=expected_columns)


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 100,
    device: str | None = None,
) -> np.ndarray:
    """Embed rationale texts with a local sentence-transformers model."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(model, device=device)
    embeddings = sentence_model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def fit_kmeans(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster rationale embeddings with k-means."""
    if len(embeddings) == 0:
        return np.asarray([], dtype=int), np.empty((0, 0), dtype=np.float32)

    from sklearn.cluster import KMeans

    cluster_count = max(1, min(int(n_clusters), len(embeddings)))
    model = KMeans(n_clusters=cluster_count, n_init="auto", random_state=random_state)
    labels = model.fit_predict(embeddings)
    return labels.astype(int), np.asarray(model.cluster_centers_, dtype=np.float32)


def project_umap(
    embeddings: np.ndarray,
    *,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
) -> np.ndarray:
    """Project embeddings into two dimensions with UMAP."""
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=np.float32)

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(int(n_neighbors), len(embeddings) - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower().strip("'") for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in STOPWORDS]


def top_terms(texts: Sequence[str], *, top_n: int = 8) -> list[str]:
    """Return the most frequent informative terms in a text collection."""
    counts = Counter()
    for text in texts:
        counts.update(_tokenize(text))
    return [term for term, _ in counts.most_common(top_n)]


def _top_categories(values: pd.Series, *, top_n: int = 3) -> str:
    counts = values.value_counts()
    parts = [f"{name} ({count})" for name, count in counts.head(top_n).items()]
    return ", ".join(parts)


def _truncate(text: str, *, length: int = 160) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= length:
        return normalized
    return normalized[: length - 3] + "..."


def build_cluster_summary(
    points: pd.DataFrame,
    embeddings: np.ndarray,
    centers: np.ndarray,
    *,
    text_column: str = "rationale",
    top_n_terms: int = 8,
    top_n_examples: int = 3,
) -> pd.DataFrame:
    """Summarize cluster size, dominant labels, keywords, and exemplars."""
    if len(points) != len(embeddings):
        raise ValueError("`points` and `embeddings` must have matching lengths.")
    if text_column not in points.columns:
        raise ValueError(f"`points` is missing text column `{text_column}`.")

    rows: list[dict[str, object]] = []
    total = len(points)
    for cluster_id in sorted(points["cluster_id"].unique()):
        cluster_mask = points["cluster_id"] == cluster_id
        cluster_points = points.loc[cluster_mask].copy()
        cluster_embeddings = embeddings[cluster_mask.to_numpy()]
        distances = np.linalg.norm(cluster_embeddings - centers[cluster_id], axis=1)
        exemplar_points = cluster_points.assign(
            distance_to_center=distances
        ).sort_values("distance_to_center")

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": f"C{int(cluster_id):02d}",
                "size": int(cluster_mask.sum()),
                "share": float(cluster_mask.sum() / total),
                "top_terms": ", ".join(
                    top_terms(cluster_points[text_column], top_n=top_n_terms)
                ),
                "top_mechanisms": _top_categories(cluster_points["mechanism_label"]),
                "top_levels": _top_categories(cluster_points["level"]),
                "top_forms": _top_categories(cluster_points["form"]),
                "example_transcript_ids": ", ".join(
                    exemplar_points["transcript_id"].astype(str).head(top_n_examples)
                ),
                "example_rationales": " || ".join(
                    exemplar_points[text_column]
                    .astype(str)
                    .head(top_n_examples)
                    .map(_truncate)
                ),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["size", "cluster_id"], ascending=[False, True]
    ).reset_index(drop=True)


def build_plot(
    points: pd.DataFrame,
    *,
    title: str,
    text_column: str = "rationale",
):
    """Create an interactive UMAP scatter plot for clustered texts."""
    plot_df = points.copy()
    if text_column not in plot_df.columns:
        raise ValueError(f"`points` is missing text column `{text_column}`.")
    plot_df["cluster_label"] = plot_df["cluster_id"].map(lambda value: f"C{value:02d}")
    hover_column = f"hover_{text_column}"
    plot_df[hover_column] = plot_df[text_column].map(_truncate)

    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        symbol="mechanism_label",
        hover_name="transcript_id",
        hover_data={
            "cluster_label": True,
            "mechanism_label": True,
            "level": True,
            "form": True,
            hover_column: True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
            text_column: False,
        },
        title=title,
        width=1200,
        height=800,
    )
    fig.update_traces(marker={"size": 8, "opacity": 0.82, "line": {"width": 0.4}})
    fig.update_layout(template="plotly_white")
    return fig


def _load_cached_embeddings(
    output_dir: Path,
    rationale_rows: pd.DataFrame,
    *,
    model: str,
    device: str | None,
) -> np.ndarray | None:
    metadata_path = output_dir / "embedding_metadata.csv"
    embeddings_path = output_dir / "embeddings.npy"
    manifest_path = output_dir / "embedding_manifest.json"

    if not (
        metadata_path.exists()
        and embeddings_path.exists()
        and manifest_path.exists()
    ):
        return None

    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    cached_pairs = list(zip(metadata["row_id"], metadata["embedding_input"]))
    current_pairs = list(
        zip(rationale_rows["row_id"], rationale_rows["embedding_input"])
    )
    if cached_pairs != current_pairs:
        return None
    if manifest.get("model") != model or manifest.get("device") != device:
        return None

    embeddings = np.load(embeddings_path)
    if len(embeddings) != len(rationale_rows):
        return None
    return embeddings


def _write_embedding_cache(
    output_dir: Path,
    rationale_rows: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    model: str,
    device: str | None,
) -> None:
    rationale_rows.loc[:, ["row_id", "embedding_input"]].to_csv(
        output_dir / "embedding_metadata.csv",
        index=False,
    )
    np.save(output_dir / "embeddings.npy", embeddings)
    with (output_dir / "embedding_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model, "device": device}, f, indent=2)
        f.write("\n")


def run_task_activity_analysis(
    results_csv: str | Path,
    *,
    output_dir: str | Path,
    mechanisms: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    forms: Sequence[str] | None = None,
    sample_size: int | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 100,
    embedding_device: str | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 7,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    force_reembed: bool = False,
) -> TaskActivityAnalysisArtifacts:
    """Run the full embedding, clustering, and UMAP pipeline for task activities."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_csv(results_csv)
    activity_rows = expand_task_activity_results(
        results,
        mechanisms=mechanisms,
        levels=levels,
        forms=forms,
    )
    if sample_size is not None:
        activity_rows = activity_rows.sample(
            n=min(sample_size, len(activity_rows)),
            random_state=random_state,
        ).sort_values(["transcript_id", "mechanism", "activity_index"])
        activity_rows.reset_index(drop=True, inplace=True)

    embeddings: np.ndarray | None = None
    if not force_reembed:
        embeddings = _load_cached_embeddings(
            resolved_output_dir,
            activity_rows,
            model=embedding_model,
            device=embedding_device,
        )

    if embeddings is None:
        embeddings = embed_texts(
            activity_rows["embedding_input"].tolist(),
            model=embedding_model,
            batch_size=embedding_batch_size,
            device=embedding_device,
        )
        _write_embedding_cache(
            resolved_output_dir,
            activity_rows,
            embeddings,
            model=embedding_model,
            device=embedding_device,
        )

    labels, centers = fit_kmeans(
        embeddings,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    coordinates = project_umap(
        embeddings,
        random_state=random_state,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
    )

    points = activity_rows.copy()
    points["cluster_id"] = labels
    points["umap_x"] = coordinates[:, 0]
    points["umap_y"] = coordinates[:, 1]
    points["cluster_label"] = points["cluster_id"].map(lambda value: f"C{value:02d}")

    summary = build_cluster_summary(
        points,
        embeddings,
        centers,
        text_column="task_activity",
    )
    title = (
        f"Task Activity Clusters ({len(points)} activities, "
        f"{points['cluster_id'].nunique()} clusters)"
    )
    figure = build_plot(points, title=title, text_column="task_activity")

    activity_rows_path = resolved_output_dir / "activity_rows.csv"
    clustered_points_path = resolved_output_dir / "clustered_activities.csv"
    cluster_summary_path = resolved_output_dir / "cluster_summary.csv"
    figure_path = resolved_output_dir / "umap_clusters.html"
    params_path = resolved_output_dir / "analysis_params.json"

    activity_rows.to_csv(activity_rows_path, index=False)
    points.to_csv(clustered_points_path, index=False)
    summary.to_csv(cluster_summary_path, index=False)
    figure.write_html(figure_path, include_plotlyjs="cdn")
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results_csv": str(results_csv),
                "output_dir": str(resolved_output_dir),
                "mechanisms": list(mechanisms or MECHANISM_KEYS),
                "levels": list(levels or DEFAULT_LEVELS),
                "forms": list(forms) if forms is not None else None,
                "sample_size": sample_size,
                "embedding_model": embedding_model,
                "embedding_batch_size": embedding_batch_size,
                "embedding_device": embedding_device,
                "n_clusters": n_clusters,
                "random_state": random_state,
                "umap_neighbors": umap_neighbors,
                "umap_min_dist": umap_min_dist,
                "force_reembed": force_reembed,
                "row_count": len(points),
            },
            f,
            indent=2,
        )
        f.write("\n")

    return TaskActivityAnalysisArtifacts(
        activity_rows_path=activity_rows_path,
        embeddings_path=resolved_output_dir / "embeddings.npy",
        clustered_points_path=clustered_points_path,
        cluster_summary_path=cluster_summary_path,
        figure_path=figure_path,
        params_path=params_path,
    )


def run_opacity_rationale_analysis(
    results_csv: str | Path,
    *,
    output_dir: str | Path,
    mechanisms: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    forms: Sequence[str] | None = None,
    sample_size: int | None = None,
    text_source: str = "rationale",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 100,
    embedding_device: str | None = None,
    n_clusters: int = 12,
    random_state: int = 7,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    force_reembed: bool = False,
) -> AnalysisArtifacts:
    """Run the full CLio-style rationale clustering pipeline."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_csv(results_csv)
    rationale_rows = expand_opacity_rationales(
        results,
        mechanisms=mechanisms,
        levels=levels,
        forms=forms,
        text_source=text_source,
    )
    if sample_size is not None:
        rationale_rows = rationale_rows.sample(
            n=min(sample_size, len(rationale_rows)),
            random_state=random_state,
        ).sort_values(["transcript_id", "mechanism"])
        rationale_rows.reset_index(drop=True, inplace=True)

    embeddings: np.ndarray | None = None
    if not force_reembed:
        embeddings = _load_cached_embeddings(
            resolved_output_dir,
            rationale_rows,
            model=embedding_model,
            device=embedding_device,
        )

    if embeddings is None:
        embeddings = embed_texts(
            rationale_rows["embedding_input"].tolist(),
            model=embedding_model,
            batch_size=embedding_batch_size,
            device=embedding_device,
        )
        _write_embedding_cache(
            resolved_output_dir,
            rationale_rows,
            embeddings,
            model=embedding_model,
            device=embedding_device,
        )

    labels, centers = fit_kmeans(
        embeddings,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    coordinates = project_umap(
        embeddings,
        random_state=random_state,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
    )

    points = rationale_rows.copy()
    points["cluster_id"] = labels
    points["umap_x"] = coordinates[:, 0]
    points["umap_y"] = coordinates[:, 1]
    points["cluster_label"] = points["cluster_id"].map(lambda value: f"C{value:02d}")

    summary = build_cluster_summary(points, embeddings, centers)
    title = (
        f"Opacity Rationale Clusters ({len(points)} rationales, "
        f"{points['cluster_id'].nunique()} clusters)"
    )
    figure = build_plot(points, title=title)

    rationale_rows_path = resolved_output_dir / "rationale_rows.csv"
    clustered_points_path = resolved_output_dir / "clustered_rationales.csv"
    cluster_summary_path = resolved_output_dir / "cluster_summary.csv"
    figure_path = resolved_output_dir / "umap_clusters.html"
    params_path = resolved_output_dir / "analysis_params.json"

    rationale_rows.to_csv(rationale_rows_path, index=False)
    points.to_csv(clustered_points_path, index=False)
    summary.to_csv(cluster_summary_path, index=False)
    figure.write_html(figure_path, include_plotlyjs="cdn")
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results_csv": str(results_csv),
                "output_dir": str(resolved_output_dir),
                "mechanisms": list(mechanisms or MECHANISM_KEYS),
                "levels": list(levels or DEFAULT_LEVELS),
                "forms": list(forms) if forms is not None else None,
                "sample_size": sample_size,
                "text_source": text_source,
                "embedding_model": embedding_model,
                "embedding_batch_size": embedding_batch_size,
                "embedding_device": embedding_device,
                "n_clusters": n_clusters,
                "random_state": random_state,
                "umap_neighbors": umap_neighbors,
                "umap_min_dist": umap_min_dist,
                "force_reembed": force_reembed,
                "row_count": len(points),
            },
            f,
            indent=2,
        )
        f.write("\n")

    return AnalysisArtifacts(
        rationale_rows_path=rationale_rows_path,
        embeddings_path=resolved_output_dir / "embeddings.npy",
        clustered_points_path=clustered_points_path,
        cluster_summary_path=cluster_summary_path,
        figure_path=figure_path,
        params_path=params_path,
    )


def run_opacity_rationale_analysis_by_mechanism(
    results_csv: str | Path,
    *,
    output_dir: str | Path,
    mechanisms: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    forms: Sequence[str] | None = None,
    sample_size: int | None = None,
    text_source: str = "rationale",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 100,
    embedding_device: str | None = None,
    n_clusters: int | None = None,
    random_state: int = 7,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    force_reembed: bool = False,
) -> dict[str, AnalysisArtifacts]:
    """Run the clustering pipeline separately for each mechanism."""
    selected_mechanisms = tuple(mechanisms or MECHANISM_KEYS)
    output_root = Path(output_dir)
    artifacts_by_mechanism: dict[str, AnalysisArtifacts] = {}

    for mechanism in selected_mechanisms:
        mechanism_dir = output_root / mechanism
        mechanism_clusters = (
            n_clusters
            if n_clusters is not None
            else DEFAULT_N_CLUSTERS
        )
        artifacts_by_mechanism[mechanism] = run_opacity_rationale_analysis(
            results_csv,
            output_dir=mechanism_dir,
            mechanisms=[mechanism],
            levels=levels,
            forms=forms,
            sample_size=sample_size,
            text_source=text_source,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_device=embedding_device,
            n_clusters=mechanism_clusters,
            random_state=random_state,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            force_reembed=force_reembed,
        )

    return artifacts_by_mechanism


def run_task_activity_analysis_by_mechanism(
    results_csv: str | Path,
    *,
    output_dir: str | Path,
    mechanisms: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    forms: Sequence[str] | None = None,
    sample_size: int | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 100,
    embedding_device: str | None = None,
    n_clusters: int | Mapping[str, int] | None = None,
    random_state: int = 7,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    force_reembed: bool = False,
) -> dict[str, TaskActivityAnalysisArtifacts]:
    """Run the activity clustering pipeline separately for each mechanism."""
    selected_mechanisms = tuple(mechanisms or MECHANISM_KEYS)
    output_root = Path(output_dir)
    artifacts_by_mechanism: dict[str, TaskActivityAnalysisArtifacts] = {}

    for mechanism in selected_mechanisms:
        mechanism_dir = output_root / mechanism
        if isinstance(n_clusters, Mapping):
            mechanism_clusters = n_clusters.get(mechanism, DEFAULT_N_CLUSTERS)
        else:
            mechanism_clusters = (
                n_clusters if n_clusters is not None else DEFAULT_N_CLUSTERS
            )
        artifacts_by_mechanism[mechanism] = run_task_activity_analysis(
            results_csv,
            output_dir=mechanism_dir,
            mechanisms=[mechanism],
            levels=levels,
            forms=forms,
            sample_size=sample_size,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_device=embedding_device,
            n_clusters=mechanism_clusters,
            random_state=random_state,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            force_reembed=force_reembed,
        )

    return artifacts_by_mechanism
