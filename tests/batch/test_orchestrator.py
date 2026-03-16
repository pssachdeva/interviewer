from pathlib import Path

import json
import pandas as pd
import pytest

from interviewer.batch import orchestrator
from interviewer.batch.models import ProviderBatchStatus, ProviderSubmission


class _FakeProvider:
    def __init__(self):
        self.downloads: list[str] = []

    def submit_batch(self, run_ctx):
        return ProviderSubmission(
            batch_id="batch-123",
            input_file_id="file-in-123",
            status="validating",
            raw_response={"id": "batch-123", "status": "validating"},
        )

    def get_batch_status(self, batch_id: str):
        return ProviderBatchStatus(
            batch_id=batch_id,
            status="completed",
            output_file_id="file-out-123",
            error_file_id="file-err-123",
            raw_response={
                "id": batch_id,
                "status": "completed",
                "output_file_id": "file-out-123",
                "error_file_id": "file-err-123",
            },
        )

    def download_file(self, file_id: str) -> str:
        self.downloads.append(file_id)
        if file_id.startswith("file-out"):
            return json.dumps(
                {
                    "custom_id": "work_0000",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": "resp_123",
                            "model": "gpt-5.4-2026-03-05",
                            "output": [
                                {"type": "reasoning", "summary": []},
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": json.dumps(
                                                {
                                                    "summary": "A summary.",
                                                    "mechanisms": {
                                                        "voice_opacity": {
                                                            "level": "none",
                                                            "form": "none",
                                                            "evidence": [],
                                                            "rationale": "No evidence.",
                                                        },
                                                        "vulnerability_opacity": {
                                                            "level": "potential",
                                                            "form": "production",
                                                            "evidence": ["Uses AI when stuck."],
                                                            "rationale": "Masks uncertainty.",
                                                        },
                                                        "provenance_opacity": {
                                                            "level": "none",
                                                            "form": "none",
                                                            "evidence": [],
                                                            "rationale": "No evidence.",
                                                        },
                                                        "attention_opacity": {
                                                            "level": "none",
                                                            "form": "none",
                                                            "evidence": [],
                                                            "rationale": "No evidence.",
                                                        },
                                                        "investment_opacity": {
                                                            "level": "none",
                                                            "form": "none",
                                                            "evidence": [],
                                                            "rationale": "No evidence.",
                                                        },
                                                    },
                                                }
                                            ),
                                        }
                                    ],
                                },
                            ],
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 20,
                                "total_tokens": 30,
                                "output_tokens_details": {"reasoning_tokens": 5},
                            },
                            "error": None,
                        },
                    },
                    "error": None,
                }
            ) + "\n"
        return f"{file_id}-content\n"


class _FakeInProgressProvider(_FakeProvider):
    def get_batch_status(self, batch_id: str):
        return ProviderBatchStatus(
            batch_id=batch_id,
            status="in_progress",
            output_file_id=None,
            error_file_id=None,
            raw_response={"id": batch_id, "status": "in_progress"},
        )


def _write_experiment(tmp_path: Path, prompt_path: Path) -> Path:
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(
        "\n".join(
            [
                "name: unit-exp",
                "provider: openai",
                "model: gpt-5.2-mini",
                f"prompt_file: {prompt_path}",
                "reasoning_effort: medium",
            ]
        ),
        encoding="utf-8",
    )
    return exp_path


def _fake_interviews() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"transcript_id": "work_0000", "text": "alpha"},
            {"transcript_id": "work_0001", "text": "beta"},
        ]
    )


def test_submit_and_collect_terminal(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system prompt", encoding="utf-8")
    exp_path = _write_experiment(tmp_path, prompt_path)

    fake_provider = _FakeProvider()
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", tmp_path / "outputs" / "runs")
    monkeypatch.setattr(orchestrator, "load_interviews", lambda split=None: _fake_interviews())
    monkeypatch.setattr(orchestrator, "_provider_for", lambda provider: fake_provider)

    manifest = orchestrator.submit_experiment(exp_path)
    run_dir = Path(manifest.run_dir)

    assert manifest.batch_id == "batch-123"
    assert manifest.transcript_count == 2
    assert not manifest.test_mode
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "input.jsonl").exists()
    assert (run_dir / "submit_response.json").exists()

    lines = (run_dir / "input.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"custom_id": "work_0000"' in lines[0]
    assert '"reasoning": {"effort": "medium"}' in lines[0]

    collect_result = orchestrator.collect_experiment(exp_path, include_transcript=True)
    assert collect_result.is_terminal
    assert collect_result.output_path is not None
    assert collect_result.error_path is not None
    assert '"custom_id": "work_0000"' in collect_result.output_path.read_text(encoding="utf-8")
    assert collect_result.error_path.read_text(encoding="utf-8") == "file-err-123-content\n"
    assert collect_result.csv_path is not None
    assert collect_result.csv_path.name == "results.csv"
    assert collect_result.csv_path.exists()
    csv_lines = collect_result.csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("transcript_id,transcript,")
    assert "alpha" in csv_lines[1]
    assert (run_dir / "status.json").exists()


def test_submit_blocks_duplicate_run(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system prompt", encoding="utf-8")
    exp_path = _write_experiment(tmp_path, prompt_path)

    monkeypatch.setattr(orchestrator, "RUNS_ROOT", tmp_path / "outputs" / "runs")
    monkeypatch.setattr(orchestrator, "load_interviews", lambda split=None: _fake_interviews())
    monkeypatch.setattr(orchestrator, "_provider_for", lambda provider: _FakeProvider())

    orchestrator.submit_experiment(exp_path)

    with pytest.raises(FileExistsError, match="Run directory already exists"):
        orchestrator.submit_experiment(exp_path)


def test_submit_test_mode_uses_first_25_and_separate_run_dir(
    tmp_path: Path, monkeypatch
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system prompt", encoding="utf-8")
    exp_path = _write_experiment(tmp_path, prompt_path)

    interviews_df = pd.DataFrame(
        [
            {"transcript_id": f"work_{i:04d}", "text": f"text-{i}"}
            for i in range(30)
        ]
    )

    monkeypatch.setattr(orchestrator, "RUNS_ROOT", tmp_path / "outputs" / "runs")
    monkeypatch.setattr(orchestrator, "load_interviews", lambda split=None: interviews_df)
    monkeypatch.setattr(orchestrator, "_provider_for", lambda provider: _FakeProvider())

    manifest = orchestrator.submit_experiment(exp_path, test_mode=True)
    run_dir = Path(manifest.run_dir)

    assert manifest.test_mode
    assert manifest.transcript_count == 25
    assert run_dir.name == "unit-exp__test"

    lines = (run_dir / "input.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 25


def test_collect_non_terminal_does_not_download(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system prompt", encoding="utf-8")
    exp_path = _write_experiment(tmp_path, prompt_path)

    monkeypatch.setattr(orchestrator, "RUNS_ROOT", tmp_path / "outputs" / "runs")
    monkeypatch.setattr(orchestrator, "load_interviews", lambda split=None: _fake_interviews())

    submit_provider = _FakeProvider()
    in_progress_provider = _FakeInProgressProvider()

    monkeypatch.setattr(orchestrator, "_provider_for", lambda provider: submit_provider)
    orchestrator.submit_experiment(exp_path)

    monkeypatch.setattr(orchestrator, "_provider_for", lambda provider: in_progress_provider)
    result = orchestrator.collect_experiment(exp_path)

    assert not result.is_terminal
    assert result.output_path is None
    assert result.error_path is None
    run_dir = Path(result.manifest.run_dir)
    assert not (run_dir / "output.jsonl").exists()
    assert not (run_dir / "error.jsonl").exists()


def test_collect_test_mode_uses_test_run_dir(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system prompt", encoding="utf-8")
    exp_path = _write_experiment(tmp_path, prompt_path)

    monkeypatch.setattr(orchestrator, "RUNS_ROOT", tmp_path / "outputs" / "runs")
    monkeypatch.setattr(orchestrator, "load_interviews", lambda split=None: _fake_interviews())

    submit_provider = _FakeProvider()
    monkeypatch.setattr(orchestrator, "_provider_for", lambda provider: submit_provider)
    manifest = orchestrator.submit_experiment(exp_path, test_mode=True)

    status_provider = _FakeInProgressProvider()
    monkeypatch.setattr(orchestrator, "_provider_for", lambda provider: status_provider)
    result = orchestrator.collect_experiment(exp_path, test_mode=True)

    assert result.manifest.run_dir == manifest.run_dir
    assert Path(result.manifest.run_dir).name == "unit-exp__test"
