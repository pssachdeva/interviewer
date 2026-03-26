"""Experiment config parsing and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from interviewer.batch.models import ExperimentConfig
from interviewer.batch.tasks import DEFAULT_TASK_TYPE, TASK_TYPE_TASK_ACTIVITY
from interviewer.opacity_columns import LEVELS, MECHANISM_KEYS

_ALLOWED_FIELDS = {
    "name",
    "provider",
    "model",
    "task_type",
    "prompt_file",
    "source_results_csv",
    "mechanisms",
    "levels",
    "reasoning_effort",
    "request",
    "metadata",
}
_DEFAULT_MODEL = "gpt-5.2-mini"
_ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be a mapping.")
    return value


def _resolve_prompt_path(prompt_file: str, experiment_path: Path) -> Path:
    candidate = Path(prompt_file)
    if candidate.is_absolute():
        return candidate

    search_paths = [
        (experiment_path.parent / candidate).resolve(),
        (experiment_path.parent.parent / candidate).resolve(),
        (Path.cwd() / candidate).resolve(),
    ]
    for path in search_paths:
        if path.exists():
            return path
    return search_paths[0]


def _require_string_list(
    name: str,
    value: Any,
    *,
    allowed_values: set[str] | None = None,
) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"`{name}` must be a list of strings when provided.")

    normalized = [item.strip() for item in value if item.strip()]
    if len(normalized) != len(value):
        raise ValueError(f"`{name}` entries must be non-empty strings.")

    if allowed_values is not None:
        invalid = [item for item in normalized if item not in allowed_values]
        if invalid:
            allowed = ", ".join(sorted(allowed_values))
            bad = ", ".join(invalid)
            raise ValueError(f"`{name}` entries must be one of: {allowed}. Got: {bad}")

    return normalized


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config YAML file."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML must parse to a mapping.")

    unknown = sorted(set(raw.keys()) - _ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"Unknown experiment field(s): {', '.join(unknown)}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`name` is required and must be a non-empty string.")

    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("`provider` is required and must be a non-empty string.")
    provider = provider.strip().lower()
    if provider != "openai":
        raise ValueError("Only `openai` provider is supported in v1.")

    model = raw.get("model", _DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("`model` must be a non-empty string.")

    prompt_file_raw = raw.get("prompt_file")
    if not isinstance(prompt_file_raw, str) or not prompt_file_raw.strip():
        raise ValueError("`prompt_file` is required and must be a non-empty string.")

    task_type = raw.get("task_type", DEFAULT_TASK_TYPE)
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("`task_type` must be a non-empty string when provided.")
    task_type = task_type.strip().lower()
    if task_type not in {DEFAULT_TASK_TYPE, TASK_TYPE_TASK_ACTIVITY}:
        raise ValueError(
            "`task_type` must be one of: "
            f"{DEFAULT_TASK_TYPE}, {TASK_TYPE_TASK_ACTIVITY}"
        )

    reasoning_effort = raw.get("reasoning_effort")
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str) or not reasoning_effort.strip():
            raise ValueError(
                "`reasoning_effort` must be a non-empty string when provided."
            )
        reasoning_effort = reasoning_effort.strip().lower()
        if reasoning_effort not in _ALLOWED_REASONING_EFFORTS:
            allowed = ", ".join(sorted(_ALLOWED_REASONING_EFFORTS))
            raise ValueError(
                f"`reasoning_effort` must be one of: {allowed}"
            )

    request = _require_mapping("request", raw.get("request"))
    metadata = _require_mapping("metadata", raw.get("metadata"))
    if reasoning_effort is not None and "reasoning" in request:
        raise ValueError(
            "Use either top-level `reasoning_effort` or `request.reasoning`, not both."
        )

    prompt_file = _resolve_prompt_path(prompt_file_raw.strip(), path)
    source_results_csv: Path | None = None
    source_results_csv_raw = raw.get("source_results_csv")
    if source_results_csv_raw is not None:
        if (
            not isinstance(source_results_csv_raw, str)
            or not source_results_csv_raw.strip()
        ):
            raise ValueError(
                "`source_results_csv` must be a non-empty string when provided."
            )
        source_results_csv = _resolve_prompt_path(source_results_csv_raw.strip(), path)

    mechanisms = _require_string_list(
        "mechanisms",
        raw.get("mechanisms"),
        allowed_values=set(MECHANISM_KEYS),
    )
    levels = _require_string_list(
        "levels",
        raw.get("levels"),
        allowed_values=set(LEVELS),
    )

    if task_type == TASK_TYPE_TASK_ACTIVITY and source_results_csv is None:
        raise ValueError(
            "`source_results_csv` is required when `task_type` is `task_activity`."
        )

    return ExperimentConfig(
        name=name.strip(),
        provider=provider,
        model=model.strip(),
        prompt_file=prompt_file,
        task_type=task_type,
        reasoning_effort=reasoning_effort,
        source_results_csv=source_results_csv,
        mechanisms=mechanisms,
        levels=levels,
        request=request,
        metadata=metadata,
        source_path=path,
    )
