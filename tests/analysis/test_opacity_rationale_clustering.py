from __future__ import annotations

import numpy as np
import pandas as pd

from interviewer.analysis.opacity_rationale_clustering import (
    build_cluster_summary,
    expand_opacity_rationales,
    run_opacity_rationale_analysis_by_mechanism,
    top_terms,
)


def test_expand_opacity_rationales_excludes_none_levels_by_default() -> None:
    results = pd.DataFrame(
        [
            {
                "transcript_id": "work_1",
                "transcript": "t1",
                "summary": "s1",
                "voice_opacity_level": "potential",
                "voice_opacity_form": "production",
                "voice_opacity_rationale": "AI smooths over how the speaker actually sounds.",
                "voice_opacity_evidence": "e1",
                "vulnerability_opacity_level": "none",
                "vulnerability_opacity_form": "none",
                "vulnerability_opacity_rationale": "No vulnerability issue.",
                "vulnerability_opacity_evidence": "",
            },
            {
                "transcript_id": "work_2",
                "transcript": "t2",
                "summary": "s2",
                "voice_opacity_level": "clear",
                "voice_opacity_form": "avoidance",
                "voice_opacity_rationale": "The speaker avoids AI in sensitive writing.",
                "voice_opacity_evidence": "e2",
                "vulnerability_opacity_level": "potential",
                "vulnerability_opacity_form": "production",
                "vulnerability_opacity_rationale": "AI helps hide uncertainty from clients.",
                "vulnerability_opacity_evidence": "e3",
            },
        ]
    )

    expanded = expand_opacity_rationales(
        results,
        mechanisms=["voice_opacity", "vulnerability_opacity"],
    )

    assert expanded["row_id"].tolist() == [
        "work_1:voice_opacity",
        "work_2:voice_opacity",
        "work_2:vulnerability_opacity",
    ]
    assert expanded["mechanism_label"].tolist() == [
        "Voice",
        "Voice",
        "Vulnerability",
    ]
    assert expanded["level"].tolist() == ["potential", "clear", "potential"]
    assert expanded["embedding_input"].tolist() == expanded["rationale"].tolist()
    assert expanded["embedding_text_source"].tolist() == ["rationale", "rationale", "rationale"]


def test_expand_opacity_rationales_can_embed_evidence() -> None:
    results = pd.DataFrame(
        [
            {
                "transcript_id": "work_1",
                "voice_opacity_level": "potential",
                "voice_opacity_form": "production",
                "voice_opacity_rationale": "AI smooths over how the speaker actually sounds.",
                "voice_opacity_evidence": "Quote one || Quote two",
            },
            {
                "transcript_id": "work_2",
                "voice_opacity_level": "clear",
                "voice_opacity_form": "avoidance",
                "voice_opacity_rationale": "The speaker avoids AI in sensitive writing.",
                "voice_opacity_evidence": "",
            },
        ]
    )

    expanded = expand_opacity_rationales(
        results,
        mechanisms=["voice_opacity"],
        text_source="evidence",
    )

    assert expanded["row_id"].tolist() == ["work_1:voice_opacity"]
    assert expanded["embedding_input"].tolist() == ["Quote one [SEP] Quote two"]
    assert expanded["embedding_text_source"].tolist() == ["evidence"]


def test_top_terms_ignores_common_words() -> None:
    terms = top_terms(
        [
            "The speaker uses AI to hide uncertainty from other people.",
            "AI helps hide uncertainty during client meetings.",
        ],
        top_n=3,
    )

    assert terms[0] == "hide"
    assert "uncertainty" in terms


def test_build_cluster_summary_reports_cluster_examples() -> None:
    points = pd.DataFrame(
        [
            {
                "transcript_id": "work_1",
                "mechanism_label": "Voice",
                "level": "potential",
                "form": "production",
                "rationale": "AI smooths tone in outgoing messages.",
                "cluster_id": 0,
            },
            {
                "transcript_id": "work_2",
                "mechanism_label": "Voice",
                "level": "clear",
                "form": "avoidance",
                "rationale": "The speaker avoids polished AI phrasing in emails.",
                "cluster_id": 0,
            },
            {
                "transcript_id": "work_3",
                "mechanism_label": "Vulnerability",
                "level": "potential",
                "form": "production",
                "rationale": "AI helps the speaker conceal uncertainty from clients.",
                "cluster_id": 1,
            },
        ]
    )
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )
    centers = np.asarray(
        [
            [0.05, 0.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )

    summary = build_cluster_summary(points, embeddings, centers, top_n_terms=3)

    assert summary["cluster_id"].tolist() == [0, 1]
    assert summary.loc[0, "size"] == 2
    assert "Voice" in summary.loc[0, "top_mechanisms"]
    assert "work_1" in summary.loc[0, "example_transcript_ids"]
    assert "uncertainty" in summary.loc[1, "top_terms"]


def test_run_analysis_by_mechanism_uses_separate_output_dirs(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(results_csv, **kwargs):
        calls.append({"results_csv": results_csv, **kwargs})
        mechanism = kwargs["mechanisms"][0]
        out_dir = kwargs["output_dir"]
        return type(
            "Artifacts",
            (),
            {
                "rationale_rows_path": out_dir / "rationale_rows.csv",
                "embeddings_path": out_dir / "embeddings.npy",
                "clustered_points_path": out_dir / "clustered_rationales.csv",
                "cluster_summary_path": out_dir / "cluster_summary.csv",
                "figure_path": out_dir / "umap_clusters.html",
                "params_path": out_dir / "analysis_params.json",
            },
        )()

    monkeypatch.setattr(
        "interviewer.analysis.opacity_rationale_clustering.run_opacity_rationale_analysis",
        fake_run,
    )

    artifacts = run_opacity_rationale_analysis_by_mechanism(
        "results.csv",
        output_dir=tmp_path,
        mechanisms=["voice_opacity", "attention_opacity"],
        levels=["potential", "clear"],
    )

    assert sorted(artifacts) == ["attention_opacity", "voice_opacity"]
    assert [call["mechanisms"] for call in calls] == [
        ["voice_opacity"],
        ["attention_opacity"],
    ]
    assert [call["output_dir"] for call in calls] == [
        tmp_path / "voice_opacity",
        tmp_path / "attention_opacity",
    ]
    assert [call["n_clusters"] for call in calls] == [10, 10]


def test_run_analysis_by_mechanism_respects_global_cluster_override(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(results_csv, **kwargs):
        calls.append({"results_csv": results_csv, **kwargs})
        out_dir = kwargs["output_dir"]
        return type(
            "Artifacts",
            (),
            {
                "rationale_rows_path": out_dir / "rationale_rows.csv",
                "embeddings_path": out_dir / "embeddings.npy",
                "clustered_points_path": out_dir / "clustered_rationales.csv",
                "cluster_summary_path": out_dir / "cluster_summary.csv",
                "figure_path": out_dir / "umap_clusters.html",
                "params_path": out_dir / "analysis_params.json",
            },
        )()

    monkeypatch.setattr(
        "interviewer.analysis.opacity_rationale_clustering.run_opacity_rationale_analysis",
        fake_run,
    )

    run_opacity_rationale_analysis_by_mechanism(
        "results.csv",
        output_dir=tmp_path,
        mechanisms=["voice_opacity", "attention_opacity"],
        n_clusters=9,
    )

    assert [call["n_clusters"] for call in calls] == [9, 9]
