"""Label rationale clusters with short phrases."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from interviewer.analysis.opacity_rationale_clustering import (
    TEXT_SOURCES,
    mechanism_label,
)

MECHANISM_BLURBS = {
    "voice_opacity": (
        "AI-mediated communication can blur personal authorship, stance, or voice "
        "in how someone comes across to other people."
    ),
    "vulnerability_opacity": (
        "AI can hide uncertainty or struggle and make someone appear more fluent or "
        "competent than they otherwise would."
    ),
    "provenance_opacity": (
        "AI can make outputs hard to explain, defend, or trace back to the user's "
        "own reasoning and understanding."
    ),
    "attention_opacity": (
        "AI can create the appearance of attention or participation without direct "
        "engagement, often through summaries or note-taking."
    ),
    "investment_opacity": (
        "AI can blur how much effort, care, or labor went into a result."
    ),
}
ALLOWED_FIELDS = {
    "name",
    "provider",
    "model",
    "prompt_file",
    "reasoning_effort",
    "input_root",
    "output_file",
    "cluster_kind",
    "text_source",
    "samples_per_cluster",
    "request",
}
ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}
CLUSTER_KINDS = {"rationale", "activity"}


@dataclass(slots=True)
class ClusterLabelConfig:
    """Config for one cluster-labeling pass."""

    name: str
    provider: str
    model: str
    prompt_file: Path
    input_root: Path
    output_file: Path
    cluster_kind: str = "rationale"
    text_source: str = "rationale"
    reasoning_effort: str | None = None
    samples_per_cluster: int = 12
    request: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def _resolve_path(raw_path: str, config_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    search_paths = [
        (config_path.parent / candidate).resolve(),
        (config_path.parent.parent / candidate).resolve(),
        (Path.cwd() / candidate).resolve(),
    ]
    for path in search_paths:
        if path.exists():
            return path
    return search_paths[0]


def load_cluster_label_config(path: str | Path) -> ClusterLabelConfig:
    """Load and validate a clustering label config."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Clustering config not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Clustering config YAML must parse to a mapping.")

    unknown = sorted(set(raw) - ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"Unknown clustering config field(s): {', '.join(unknown)}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`name` is required and must be a non-empty string.")

    provider = raw.get("provider")
    if not isinstance(provider, str) or provider.strip().lower() != "openai":
        raise ValueError("`provider` must be `openai`.")

    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("`model` is required and must be a non-empty string.")

    prompt_file_raw = raw.get("prompt_file")
    if not isinstance(prompt_file_raw, str) or not prompt_file_raw.strip():
        raise ValueError("`prompt_file` is required and must be a non-empty string.")

    input_root_raw = raw.get("input_root")
    if not isinstance(input_root_raw, str) or not input_root_raw.strip():
        raise ValueError("`input_root` is required and must be a non-empty string.")

    reasoning_effort = raw.get("reasoning_effort")
    if reasoning_effort is not None:
        if (
            not isinstance(reasoning_effort, str)
            or reasoning_effort not in ALLOWED_REASONING_EFFORTS
        ):
            allowed = ", ".join(sorted(ALLOWED_REASONING_EFFORTS))
            raise ValueError(f"`reasoning_effort` must be one of: {allowed}")

    samples_per_cluster = raw.get("samples_per_cluster", 12)
    if not isinstance(samples_per_cluster, int) or samples_per_cluster <= 0:
        raise ValueError("`samples_per_cluster` must be a positive integer.")

    request = raw.get("request") or {}
    if not isinstance(request, dict):
        raise ValueError("`request` must be a mapping when provided.")

    prompt_file = _resolve_path(prompt_file_raw.strip(), resolved)
    input_root = _resolve_path(input_root_raw.strip(), resolved)
    cluster_kind = raw.get("cluster_kind", "rationale")
    if not isinstance(cluster_kind, str) or cluster_kind not in CLUSTER_KINDS:
        allowed = ", ".join(sorted(CLUSTER_KINDS))
        raise ValueError(f"`cluster_kind` must be one of: {allowed}")
    text_source = raw.get("text_source", "rationale")
    if cluster_kind == "rationale":
        if not isinstance(text_source, str) or text_source not in TEXT_SOURCES:
            allowed = ", ".join(TEXT_SOURCES)
            raise ValueError(f"`text_source` must be one of: {allowed}")
    else:
        text_source = "task_activity"
    output_file_raw = raw.get("output_file")
    if output_file_raw is None:
        output_file = input_root / "cluster_labels.csv"
    else:
        if not isinstance(output_file_raw, str) or not output_file_raw.strip():
            raise ValueError("`output_file` must be a non-empty string when provided.")
        output_file = _resolve_path(output_file_raw.strip(), resolved)

    return ClusterLabelConfig(
        name=name.strip(),
        provider="openai",
        model=model.strip(),
        prompt_file=prompt_file,
        input_root=input_root,
        output_file=output_file,
        cluster_kind=cluster_kind,
        text_source=text_source,
        reasoning_effort=reasoning_effort,
        samples_per_cluster=samples_per_cluster,
        request=request,
        source_path=resolved,
    )


def _available_mechanism_dirs(input_root: Path) -> list[Path]:
    return _available_mechanism_dirs_for(
        input_root,
        clustered_filename="clustered_rationales.csv",
    )


def _available_mechanism_dirs_for(
    input_root: Path,
    *,
    clustered_filename: str,
) -> list[Path]:
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir()
        and (path / clustered_filename).exists()
        and (path / "embeddings.npy").exists()
    )


def _format_counts(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _representative_rows(
    cluster_df: pd.DataFrame,
    cluster_embeddings: np.ndarray,
    *,
    sample_count: int,
) -> pd.DataFrame:
    center = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - center, axis=1)
    ranked = cluster_df.assign(distance_to_center=distances).sort_values(
        ["distance_to_center", "transcript_id"]
    )
    return ranked.head(min(sample_count, len(ranked))).copy()


def _render_representative_snippets(rows: pd.DataFrame, *, text_source: str) -> str:
    rendered: list[str] = []
    for index, (_, row) in enumerate(rows.iterrows(), start=1):
        rendered.append(
            f"{index}. [{row['transcript_id']}] "
            f"level={row['level']}; form={row['form']}; "
            f"{text_source}={row[text_source]}"
        )
    return "\n".join(rendered)


def snippet_source_label(text_source: str) -> str:
    if text_source == "evidence":
        return "Quoted evidence excerpts selected for the mechanism"
    if text_source == "task_activity":
        return "Short task-activity clauses assigned to the mechanism"
    return "Model-written coding rationales for the mechanism"


def build_cluster_label_rows(config: ClusterLabelConfig) -> pd.DataFrame:
    """Prepare one row per mechanism/cluster for downstream labeling."""
    clustered_filename = (
        "clustered_activities.csv"
        if config.cluster_kind == "activity"
        else "clustered_rationales.csv"
    )
    mechanism_dirs = _available_mechanism_dirs_for(
        config.input_root,
        clustered_filename=clustered_filename,
    )
    if not mechanism_dirs:
        raise FileNotFoundError(
            f"No clustering artifacts found under {config.input_root}."
        )

    rows: list[dict[str, Any]] = []
    for mechanism_dir in mechanism_dirs:
        mechanism = mechanism_dir.name
        points = pd.read_csv(mechanism_dir / clustered_filename)
        embeddings = np.load(mechanism_dir / "embeddings.npy")
        if len(points) != len(embeddings):
            raise ValueError(
                f"Embedding count does not match rationale rows for {mechanism}."
            )

        for cluster_id in sorted(points["cluster_id"].unique()):
            cluster_mask = points["cluster_id"] == cluster_id
            cluster_points = points.loc[cluster_mask].copy()
            cluster_embeddings = embeddings[cluster_mask.to_numpy()]
            representative = _representative_rows(
                cluster_points,
                cluster_embeddings,
                sample_count=config.samples_per_cluster,
            )

            rows.append(
                {
                    "mechanism": mechanism,
                    "mechanism_label": mechanism_label(mechanism),
                    "mechanism_blurb": MECHANISM_BLURBS[mechanism],
                    "text_source": config.text_source,
                    "snippet_source_label": snippet_source_label(config.text_source),
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(cluster_mask.sum()),
                    "level_counts": _format_counts(cluster_points["level"]),
                    "form_counts": _format_counts(cluster_points["form"]),
                    "sample_transcript_ids": " || ".join(
                        representative["transcript_id"].astype(str)
                    ),
                    "representative_snippets": _render_representative_snippets(
                        representative,
                        text_source=config.text_source,
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["mechanism", "cluster_id"]
    ).reset_index(drop=True)


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    response_dict = response.model_dump() if hasattr(response, "model_dump") else None

    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
            refusal = getattr(content, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                return refusal.strip()
            if isinstance(content, dict):
                dict_text = content.get("text")
                if isinstance(dict_text, str) and dict_text.strip():
                    return dict_text.strip()
                dict_refusal = content.get("refusal")
                if isinstance(dict_refusal, str) and dict_refusal.strip():
                    return dict_refusal.strip()

    if isinstance(response_dict, dict):
        for output in response_dict.get("output", []) or []:
            for content in output.get("content", []) or []:
                if not isinstance(content, dict):
                    continue
                dict_text = content.get("text")
                if isinstance(dict_text, str) and dict_text.strip():
                    return dict_text.strip()
                dict_refusal = content.get("refusal")
                if isinstance(dict_refusal, str) and dict_refusal.strip():
                    return dict_refusal.strip()
                if content.get("type") == "output_text":
                    text_payload = content.get("text")
                    if isinstance(text_payload, str) and text_payload.strip():
                        return text_payload.strip()

    raise ValueError("Response did not contain output text.")


def _label_only(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned.strip().strip('"').strip("'")


def _client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def label_clusters(config: ClusterLabelConfig) -> pd.DataFrame:
    """Label all discovered clusters and write one combined output file."""
    prompt_template = config.prompt_file.read_text(encoding="utf-8")
    prepared = build_cluster_label_rows(config)
    client = _client()

    output_rows: list[dict[str, Any]] = []
    iterator = tqdm(
        prepared.iterrows(),
        total=len(prepared),
        desc="Labeling clusters",
        unit="cluster",
    )
    for _, row in iterator:
        prompt_text = prompt_template.format(
            mechanism_label=row["mechanism_label"],
            mechanism_blurb=row["mechanism_blurb"],
            snippet_source_label=row["snippet_source_label"],
            cluster_id=row["cluster_id"],
            cluster_size=row["cluster_size"],
            level_counts=row["level_counts"],
            form_counts=row["form_counts"],
            representative_snippets=row["representative_snippets"],
        )

        request_kwargs = dict(config.request)
        iterator.set_postfix(
            mechanism=row["mechanism"],
            cluster=int(row["cluster_id"]),
            refresh=False,
        )

        base_output = {
            **row.to_dict(),
            "prompt_text": prompt_text,
            "cluster_label_phrase": "",
            "raw_output_text": "",
            "response_json": "",
            "error": "",
        }
        response = None

        try:
            response = client.responses.create(
                model=config.model,
                input=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": "Provide the cluster label."},
                ],
                reasoning=(
                    {"effort": config.reasoning_effort}
                    if config.reasoning_effort
                    else None
                ),
                **request_kwargs,
            )
            raw_text = _extract_response_text(response)
            output_rows.append(
                {
                    **base_output,
                    "cluster_label_phrase": _label_only(raw_text),
                    "raw_output_text": raw_text,
                    "response_json": json.dumps(
                        response.model_dump(),
                        ensure_ascii=False,
                    ),
                }
            )
        except Exception as exc:
            response_json = ""
            if response is not None:
                try:
                    response_json = json.dumps(
                        response.model_dump(),
                        ensure_ascii=False,
                    )
                except Exception:
                    response_json = ""
            output_rows.append(
                {
                    **base_output,
                    "response_json": response_json,
                    "error": str(exc),
                }
            )

        interim_df = pd.DataFrame(output_rows)
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        interim_df.to_csv(config.output_file, index=False)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(config.output_file, index=False)
    return output_df
