"""Export raw batch outputs into task-specific flat CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from interviewer.batch.config import load_experiment_config
from interviewer.batch.orchestrator import _run_dir_for
from interviewer.batch.tasks import (
    flatten_opacity_output_record,
    flatten_task_activity_output_record,
    get_task_adapter,
)


def flatten_output_record(record: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible flattener for opacity-coding outputs."""
    return flatten_opacity_output_record(record)


def export_results_csv(
    experiment_path: str | Path,
    test_mode: bool = False,
    include_transcript: bool = False,
    output_path: str | Path | None = None,
) -> Path:
    """Export batch JSONL outputs for an experiment into a flat CSV file."""
    experiment = load_experiment_config(experiment_path)
    run_dir = _run_dir_for(experiment.name, test_mode=test_mode)
    adapter = get_task_adapter(experiment.task_type)
    return adapter.export_results(
        experiment=experiment,
        run_dir=run_dir,
        include_transcript=include_transcript,
        output_path=output_path,
    )


__all__ = [
    "export_results_csv",
    "flatten_output_record",
    "flatten_opacity_output_record",
    "flatten_task_activity_output_record",
]
