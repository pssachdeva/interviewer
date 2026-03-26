"""Orchestrates submit and collect workflows for batch experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from interviewer.batch.config import load_experiment_config
from interviewer.batch.formatting import write_jsonl
from interviewer.batch.models import (
    TERMINAL_BATCH_STATUSES,
    CollectResult,
    ProviderBatchStatus,
    RunContext,
    RunManifest,
)
from interviewer.batch.providers.base import ProviderClient
from interviewer.batch.providers.openai_provider import OpenAIProvider
from interviewer.batch.tasks import get_task_adapter

RUNS_ROOT = Path("outputs") / "runs"
MANIFEST_FILENAME = "manifest.json"
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    _json_dump(path, manifest.to_dict())


def _read_manifest(path: Path) -> RunManifest:
    with path.open("r", encoding="utf-8") as f:
        return RunManifest.from_dict(json.load(f))


def _provider_for(provider: str) -> ProviderClient:
    if provider == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unsupported provider: {provider}")


def _run_dir_for(experiment_name: str, test_mode: bool = False) -> Path:
    suffix = "__test" if test_mode else ""
    return RUNS_ROOT / f"{experiment_name}{suffix}"


def submit_experiment(path: str | Path, test_mode: bool = False) -> RunManifest:
    """Submit one batch job for an experiment config."""
    experiment = load_experiment_config(path)

    if not experiment.prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {experiment.prompt_file}")

    run_dir = _run_dir_for(experiment.name, test_mode=test_mode)
    manifest_path = run_dir / MANIFEST_FILENAME
    if run_dir.exists():
        raise FileExistsError(
            "Run directory already exists for experiment "
            f"`{experiment.name}`: {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=False)

    prompt_text = experiment.prompt_file.read_text(encoding="utf-8")
    task_adapter = get_task_adapter(experiment.task_type)
    input_path = run_dir / "input.jsonl"
    rows, item_count = task_adapter.build_batch_rows(
        experiment=experiment,
        prompt_text=prompt_text,
        test_mode=test_mode,
    )
    write_jsonl(input_path, rows)

    manifest = RunManifest(
        experiment_name=experiment.name,
        experiment_path=str(experiment.source_path or Path(path).resolve()),
        provider=experiment.provider,
        model=experiment.model,
        prompt_file=str(experiment.prompt_file),
        prompt_sha256=_sha256_text(prompt_text),
        experiment_sha256=_sha256_file(experiment.source_path or Path(path).resolve()),
        created_at_utc=_now_utc_iso(),
        run_dir=str(run_dir.resolve()),
        input_jsonl_path=str(input_path.resolve()),
        test_mode=test_mode,
        transcript_count=item_count,
        status="prepared",
    )
    _write_manifest(manifest_path, manifest)

    provider = _provider_for(experiment.provider)
    submission = provider.submit_batch(
        RunContext(experiment=experiment, input_jsonl_path=input_path)
    )

    _json_dump(run_dir / "submit_response.json", submission.raw_response)

    manifest.batch_id = submission.batch_id
    manifest.input_file_id = submission.input_file_id
    manifest.status = submission.status
    _write_manifest(manifest_path, manifest)

    return manifest


def collect_experiment(
    path: str | Path,
    test_mode: bool = False,
    include_transcript: bool = False,
) -> CollectResult:
    """Fetch status and download terminal outputs for an experiment run."""
    experiment = load_experiment_config(path)
    run_dir = _run_dir_for(experiment.name, test_mode=test_mode)
    manifest_path = run_dir / MANIFEST_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Run manifest not found for experiment "
            f"`{experiment.name}`: {manifest_path}"
        )

    manifest = _read_manifest(manifest_path)
    if not manifest.batch_id:
        raise ValueError("Manifest has no batch_id; submit may have failed.")

    provider = _provider_for(manifest.provider)
    status: ProviderBatchStatus = provider.get_batch_status(manifest.batch_id)

    status_path = run_dir / "status.json"
    _json_dump(status_path, status.raw_response)

    manifest.status = status.status
    manifest.output_file_id = status.output_file_id
    manifest.error_file_id = status.error_file_id

    output_path: Path | None = None
    error_path: Path | None = None
    csv_path: Path | None = None

    if status.status in TERMINAL_BATCH_STATUSES:
        if status.output_file_id:
            output_path = run_dir / "output.jsonl"
            output_path.write_text(
                provider.download_file(status.output_file_id),
                encoding="utf-8",
            )

        if status.error_file_id:
            error_path = run_dir / "error.jsonl"
            error_path.write_text(
                provider.download_file(status.error_file_id),
                encoding="utf-8",
            )

        manifest.collected_at_utc = _now_utc_iso()

        existing_output_path = run_dir / "output.jsonl"
        if existing_output_path.exists():
            from interviewer.batch.export import export_results_csv

            csv_path = export_results_csv(
                path,
                test_mode=test_mode,
                include_transcript=include_transcript,
            )

    _write_manifest(manifest_path, manifest)

    return CollectResult(
        manifest=manifest,
        status=status,
        status_path=status_path,
        output_path=output_path,
        error_path=error_path,
        csv_path=csv_path,
    )
