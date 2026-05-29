from __future__ import annotations

import numpy as np
import pandas as pd

from interviewer.clustering.cluster_labeling import (
    ClusterLabelConfig,
    build_cluster_label_rows,
    load_cluster_label_config,
)


def test_load_cluster_label_config_success(tmp_path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    prompt_file = prompts / "cluster_prompt.txt"
    prompt_file.write_text("Prompt {mechanism_label}", encoding="utf-8")

    input_root = tmp_path / "artifacts"
    input_root.mkdir()

    config_path = tmp_path / "cluster.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: labels",
                "provider: openai",
                "model: gpt-5.4",
                "prompt_file: prompts/cluster_prompt.txt",
                "reasoning_effort: medium",
                "input_root: artifacts",
                "output_file: outputs/labels.csv",
                "samples_per_cluster: 5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_cluster_label_config(config_path)

    assert config.name == "labels"
    assert config.prompt_file == prompt_file.resolve()
    assert config.input_root == input_root.resolve()
    assert config.output_file == (tmp_path / "outputs" / "labels.csv").resolve()
    assert config.cluster_kind == "rationale"
    assert config.text_source == "rationale"
    assert config.reasoning_effort == "medium"
    assert config.samples_per_cluster == 5


def test_load_cluster_label_config_defaults_output_next_to_input_root(tmp_path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    prompt_file = prompts / "cluster_prompt.txt"
    prompt_file.write_text("Prompt {mechanism_label}", encoding="utf-8")

    input_root = tmp_path / "artifacts"
    input_root.mkdir()

    config_path = tmp_path / "cluster.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: labels",
                "provider: openai",
                "model: gpt-5.4",
                "prompt_file: prompts/cluster_prompt.txt",
                "input_root: artifacts",
            ]
        ),
        encoding="utf-8",
    )

    config = load_cluster_label_config(config_path)

    assert config.output_file == (input_root / "cluster_labels.csv").resolve()


def test_build_cluster_label_rows_uses_centroid_nearest_examples(tmp_path) -> None:
    input_root = tmp_path / "artifacts"
    mechanism_dir = input_root / "voice_opacity"
    mechanism_dir.mkdir(parents=True)

    points = pd.DataFrame(
        [
            {
                "transcript_id": "work_0001",
                "cluster_id": 0,
                "level": "none",
                "form": "none",
                "rationale": "No voice issue appears in the transcript.",
                "evidence": "Evidence A",
            },
            {
                "transcript_id": "work_0002",
                "cluster_id": 0,
                "level": "clear",
                "form": "avoidance",
                "rationale": "The speaker avoids AI rewriting to preserve voice.",
                "evidence": "Evidence B",
            },
            {
                "transcript_id": "work_0003",
                "cluster_id": 0,
                "level": "potential",
                "form": "production",
                "rationale": "AI drafting could flatten personal tone.",
                "evidence": "Evidence C",
            },
            {
                "transcript_id": "work_0004",
                "cluster_id": 1,
                "level": "clear",
                "form": "production",
                "rationale": "Clients cannot tell what the speaker actually wrote.",
                "evidence": "Evidence D",
            },
        ]
    )
    points.to_csv(mechanism_dir / "clustered_rationales.csv", index=False)
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
        ],
        dtype=np.float32,
    )
    np.save(mechanism_dir / "embeddings.npy", embeddings)

    config = ClusterLabelConfig(
        name="labels",
        provider="openai",
        model="gpt-5.4",
        prompt_file=tmp_path / "prompt.txt",
        input_root=input_root,
        output_file=tmp_path / "labels.csv",
        samples_per_cluster=2,
    )

    rows = build_cluster_label_rows(config)

    first_cluster = rows.iloc[0]
    assert first_cluster["mechanism"] == "voice_opacity"
    assert first_cluster["cluster_id"] == 0
    assert first_cluster["cluster_size"] == 3
    assert "work_0001" in first_cluster["sample_transcript_ids"]
    assert "work_0002" in first_cluster["sample_transcript_ids"]
    assert "work_0003" not in first_cluster["sample_transcript_ids"]
    assert first_cluster["text_source"] == "rationale"
    assert (
        "rationale=The speaker avoids AI rewriting to preserve voice."
        in first_cluster["representative_snippets"]
    )


def test_build_cluster_label_rows_can_use_evidence(tmp_path) -> None:
    input_root = tmp_path / "artifacts"
    mechanism_dir = input_root / "voice_opacity"
    mechanism_dir.mkdir(parents=True)

    points = pd.DataFrame(
        [
            {
                "transcript_id": "work_0001",
                "cluster_id": 0,
                "level": "clear",
                "form": "avoidance",
                "rationale": "Rationale A",
                "evidence": "Quoted evidence A",
            },
            {
                "transcript_id": "work_0002",
                "cluster_id": 0,
                "level": "potential",
                "form": "production",
                "rationale": "Rationale B",
                "evidence": "Quoted evidence B",
            },
        ]
    )
    points.to_csv(mechanism_dir / "clustered_rationales.csv", index=False)
    np.save(
        mechanism_dir / "embeddings.npy",
        np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
    )

    config = ClusterLabelConfig(
        name="labels",
        provider="openai",
        model="gpt-5.4",
        prompt_file=tmp_path / "prompt.txt",
        input_root=input_root,
        output_file=tmp_path / "labels.csv",
        text_source="evidence",
        samples_per_cluster=2,
    )

    rows = build_cluster_label_rows(config)

    assert rows.loc[0, "text_source"] == "evidence"
    assert (
        rows.loc[0, "snippet_source_label"]
        == "Quoted evidence excerpts selected for the mechanism"
    )
    assert "evidence=Quoted evidence A" in rows.loc[0, "representative_snippets"]


def test_load_cluster_label_config_supports_activity_clusters(tmp_path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    prompt_file = prompts / "cluster_prompt.txt"
    prompt_file.write_text("Prompt {mechanism_label}", encoding="utf-8")

    input_root = tmp_path / "artifacts"
    input_root.mkdir()

    config_path = tmp_path / "cluster.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: labels",
                "provider: openai",
                "model: gpt-5.4",
                "prompt_file: prompts/cluster_prompt.txt",
                "input_root: artifacts",
                "cluster_kind: activity",
            ]
        ),
        encoding="utf-8",
    )

    config = load_cluster_label_config(config_path)

    assert config.cluster_kind == "activity"
    assert config.text_source == "task_activity"


def test_build_cluster_label_rows_supports_activity_clusters(tmp_path) -> None:
    input_root = tmp_path / "artifacts"
    mechanism_dir = input_root / "voice_opacity"
    mechanism_dir.mkdir(parents=True)

    points = pd.DataFrame(
        [
            {
                "transcript_id": "work_0001",
                "cluster_id": 0,
                "level": "clear",
                "form": "production",
                "task_activity": "drafting and revising client emails",
            },
            {
                "transcript_id": "work_0002",
                "cluster_id": 0,
                "level": "potential",
                "form": "mixed",
                "task_activity": "writing customer-facing emails and letters",
            },
        ]
    )
    points.to_csv(mechanism_dir / "clustered_activities.csv", index=False)
    np.save(
        mechanism_dir / "embeddings.npy",
        np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
    )

    config = ClusterLabelConfig(
        name="labels",
        provider="openai",
        model="gpt-5.4",
        prompt_file=tmp_path / "prompt.txt",
        input_root=input_root,
        output_file=tmp_path / "labels.csv",
        cluster_kind="activity",
        text_source="task_activity",
        samples_per_cluster=2,
    )

    rows = build_cluster_label_rows(config)

    assert rows.loc[0, "text_source"] == "task_activity"
    assert (
        rows.loc[0, "snippet_source_label"]
        == "Short task-activity clauses assigned to the mechanism"
    )
    assert (
        "task_activity=drafting and revising client emails"
        in rows.loc[0, "representative_snippets"]
    )
