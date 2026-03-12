from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from interviewer.batch.models import CollectResult, ProviderBatchStatus, RunManifest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_submit_cli_handles_missing_file(capsys) -> None:
    module = _load_module(REPO_ROOT / "scripts" / "submit_experiment.py", "submit_cli_1")
    rc = module.main(["does-not-exist.yaml"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "Error:" in captured.err


def test_submit_cli_duplicate_run_error(monkeypatch, capsys) -> None:
    module = _load_module(REPO_ROOT / "scripts" / "submit_experiment.py", "submit_cli_2")

    def _raise_duplicate(_, test_mode=False):
        raise FileExistsError("duplicate")

    monkeypatch.setattr(module, "submit_experiment", _raise_duplicate)
    rc = module.main(["experiments/example_openai_batch.yaml"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "duplicate" in captured.err


def test_submit_cli_missing_env_error(monkeypatch, capsys) -> None:
    module = _load_module(REPO_ROOT / "scripts" / "submit_experiment.py", "submit_cli_3")

    def _raise_env(_, test_mode=False):
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")

    monkeypatch.setattr(module, "submit_experiment", _raise_env)
    rc = module.main(["experiments/example_openai_batch.yaml"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "OPENAI_API_KEY" in captured.err


def test_submit_cli_test_flag_passes_through(monkeypatch, capsys) -> None:
    module = _load_module(REPO_ROOT / "scripts" / "submit_experiment.py", "submit_cli_4")

    calls: list[tuple[str, bool]] = []

    def _fake_submit(experiment, test_mode=False):
        calls.append((str(experiment), test_mode))
        return RunManifest(
            experiment_name="exp",
            experiment_path=str(experiment),
            provider="openai",
            model="gpt-5.4",
            prompt_file="prompts/prompt.txt",
            prompt_sha256="x",
            experiment_sha256="y",
            created_at_utc="2026-01-01T00:00:00+00:00",
            run_dir="outputs/runs/exp__test",
            input_jsonl_path="outputs/runs/exp__test/input.jsonl",
            test_mode=True,
            transcript_count=25,
            batch_id="batch-1",
            input_file_id="file-1",
            status="validating",
        )

    monkeypatch.setattr(module, "submit_experiment", _fake_submit)
    rc = module.main(["--test", "experiments/exp0.0.yaml"])
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == [("experiments/exp0.0.yaml", True)]
    assert "Transcript count: 25" in captured.out


def test_collect_cli_non_terminal(monkeypatch, capsys) -> None:
    module = _load_module(REPO_ROOT / "scripts" / "collect_batch_results.py", "collect_cli_1")

    manifest = RunManifest(
        experiment_name="exp",
        experiment_path="experiments/exp.yaml",
        provider="openai",
        model="gpt-5.2-mini",
        prompt_file="prompts/prompt.txt",
        prompt_sha256="x",
        experiment_sha256="y",
        created_at_utc="2026-01-01T00:00:00+00:00",
        run_dir="outputs/runs/exp",
        input_jsonl_path="outputs/runs/exp/input.jsonl",
        batch_id="batch-1",
        input_file_id="file-1",
        status="in_progress",
    )
    status = ProviderBatchStatus(batch_id="batch-1", status="in_progress")
    fake_result = CollectResult(
        manifest=manifest,
        status=status,
        status_path=Path("outputs/runs/exp/status.json"),
        output_path=None,
        error_path=None,
        csv_path=None,
    )

    monkeypatch.setattr(module, "collect_experiment", lambda _, test_mode=False: fake_result)

    rc = module.main(["experiments/example_openai_batch.yaml"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "in_progress" in captured.out
    assert "not in terminal state" in captured.out


def test_collect_cli_test_flag_passes_through(monkeypatch, capsys) -> None:
    module = _load_module(REPO_ROOT / "scripts" / "collect_batch_results.py", "collect_cli_2")

    calls: list[tuple[str, bool]] = []

    manifest = RunManifest(
        experiment_name="exp",
        experiment_path="experiments/exp.yaml",
        provider="openai",
        model="gpt-5.4",
        prompt_file="prompts/prompt.txt",
        prompt_sha256="x",
        experiment_sha256="y",
        created_at_utc="2026-01-01T00:00:00+00:00",
        run_dir="outputs/runs/exp__test",
        input_jsonl_path="outputs/runs/exp__test/input.jsonl",
        test_mode=True,
        transcript_count=25,
        batch_id="batch-1",
        input_file_id="file-1",
        status="in_progress",
    )
    status = ProviderBatchStatus(batch_id="batch-1", status="in_progress")
    fake_result = CollectResult(
        manifest=manifest,
        status=status,
        status_path=Path("outputs/runs/exp__test/status.json"),
        output_path=None,
        error_path=None,
        csv_path=Path("outputs/runs/exp__test/results.csv"),
    )

    def _fake_collect(experiment, test_mode=False):
        calls.append((str(experiment), test_mode))
        return fake_result

    monkeypatch.setattr(module, "collect_experiment", _fake_collect)

    rc = module.main(["--test", "experiments/exp0.0.yaml"])
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == [("experiments/exp0.0.yaml", True)]
    assert "CSV file:" in captured.out
