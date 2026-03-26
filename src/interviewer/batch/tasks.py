"""Task adapters for batch submission and result export."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from interviewer.batch.formatting import (
    build_batch_line_from_messages,
    iter_batch_lines,
)
from interviewer.data import load_interviews
from interviewer.opacity_columns import (
    BASE_RESULTS_COLUMNS,
    MECHANISM_KEYS,
    evidence_column,
    form_column,
    level_column,
    rationale_column,
)

TASK_TYPE_OPACITY_CODING = "opacity_coding"
TASK_TYPE_TASK_ACTIVITY = "task_activity"
DEFAULT_TASK_TYPE = TASK_TYPE_OPACITY_CODING
TEST_ITEM_LIMIT = 25

TASK_ACTIVITY_RESULTS_COLUMNS = [
    "transcript_id",
    "mechanism",
    "level",
    "form",
    "rationale",
    "evidence",
    "task_activities",
    "response_status_code",
    "response_id",
    "model",
    "parse_error",
    "raw_output_text",
    "error",
    "response_error",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
]

TASK_ACTIVITY_MECHANISM_CONTEXTS = {
    "voice_opacity": (
        "This concerns cases where AI shapes how a person comes across to others in "
        "communication, making voice, authorship, stance, or personal expression "
        "harder to read."
    ),
    "vulnerability_opacity": (
        "This concerns cases where AI can hide uncertainty, struggle, or limits and "
        "make someone appear more fluent or capable than they otherwise would."
    ),
    "provenance_opacity": (
        "This concerns cases where AI helps produce an artifact whose basis, "
        "reasoning, or authorship becomes harder for other people to trace, explain, "
        "defend, or hold accountable."
    ),
    "attention_opacity": (
        "This concerns cases where AI creates the appearance of attention, "
        "participation, or engagement without direct involvement."
    ),
    "investment_opacity": (
        "This concerns cases where AI makes the amount of effort, care, or labor "
        "behind an output harder for other people to judge."
    ),
}


class BatchTaskAdapter(Protocol):
    """Task-specific input building and result export hooks."""

    def build_batch_rows(
        self,
        *,
        experiment: Any,
        prompt_text: str,
        test_mode: bool,
    ) -> tuple[list[dict[str, Any]], int]: ...

    def export_results(
        self,
        *,
        experiment: Any,
        run_dir: Path,
        include_transcript: bool,
        output_path: str | Path | None = None,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class _OpacityCodingAdapter:
    def build_batch_rows(
        self,
        *,
        experiment: Any,
        prompt_text: str,
        test_mode: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        interviews_df = load_interviews()
        if test_mode:
            interviews_df = interviews_df.head(TEST_ITEM_LIMIT).copy()

        rows = list(
            iter_batch_lines(
                interviews_df=interviews_df,
                system_prompt=prompt_text,
                model=experiment.model,
                reasoning_effort=experiment.reasoning_effort,
                request_overrides=experiment.request,
            )
        )
        return rows, len(interviews_df)

    def export_results(
        self,
        *,
        experiment: Any,
        run_dir: Path,
        include_transcript: bool,
        output_path: str | Path | None = None,
    ) -> Path:
        return _export_opacity_coding_results_csv(
            run_dir=run_dir,
            include_transcript=include_transcript,
            output_path=output_path,
        )


@dataclass(frozen=True, slots=True)
class _TaskActivityAdapter:
    def build_batch_rows(
        self,
        *,
        experiment: Any,
        prompt_text: str,
        test_mode: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        source_df = _load_task_activity_source_rows(experiment)
        if test_mode:
            source_df = source_df.head(TEST_ITEM_LIMIT).copy()

        rows: list[dict[str, Any]] = []
        for _, row in source_df.iterrows():
            user_message = prompt_text.format(
                mechanism=_string_value(row["mechanism_label"]),
                mechanism_context=_string_value(row["mechanism_context"]),
                level=_string_value(row["level"]),
                form=_string_value(row["form"]),
                rationale=_string_value(row["rationale"]),
                evidence=_string_value(row["evidence"]),
                transcript=_string_value(row["transcript"]),
            )
            rows.append(
                build_batch_line_from_messages(
                    custom_id=_string_value(row["custom_id"]),
                    system_prompt="",
                    user_message=user_message,
                    model=experiment.model,
                    reasoning_effort=experiment.reasoning_effort,
                    request_overrides=experiment.request,
                )
            )

        return rows, len(source_df)

    def export_results(
        self,
        *,
        experiment: Any,
        run_dir: Path,
        include_transcript: bool,
        output_path: str | Path | None = None,
    ) -> Path:
        return _export_task_activity_results_csv(
            experiment=experiment,
            run_dir=run_dir,
            include_transcript=include_transcript,
            output_path=output_path,
        )


_ADAPTERS: dict[str, BatchTaskAdapter] = {
    TASK_TYPE_OPACITY_CODING: _OpacityCodingAdapter(),
    TASK_TYPE_TASK_ACTIVITY: _TaskActivityAdapter(),
}


def get_task_adapter(task_type: str) -> BatchTaskAdapter:
    """Return the registered adapter for one task type."""
    try:
        return _ADAPTERS[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type: {task_type}") from exc


def _string_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _extract_output_text(record: dict[str, Any]) -> str | None:
    response = record.get("response") or {}
    body = response.get("body") or {}
    outputs = body.get("output") or []

    for output in outputs:
        if output.get("type") != "message":
            continue

        for content in output.get("content") or []:
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text

    return None


def _extract_user_message(input_record: dict[str, Any]) -> str:
    body = input_record.get("body") or {}
    inputs = body.get("input") or []
    for item in inputs:
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
    return ""


def _load_transcript_map(input_jsonl_path: Path) -> dict[str, str]:
    transcript_map: dict[str, str] = {}
    if not input_jsonl_path.exists():
        return transcript_map

    with input_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            custom_id = record.get("custom_id")
            if isinstance(custom_id, str):
                transcript_map[custom_id] = _extract_user_message(record)

    return transcript_map


def _join_evidence(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return " || ".join(str(item) for item in value)


def _coerce_level(mechanism: dict[str, Any]) -> str:
    level = mechanism.get("level")
    if isinstance(level, str):
        return level

    present = mechanism.get("present")
    if present is True:
        return "clear"
    if present is False:
        return "none"

    return ""


def _coerce_form(mechanism: dict[str, Any], level: str) -> str:
    form = mechanism.get("form")
    if isinstance(form, str):
        return form
    if level == "none":
        return "none"
    return ""


def _load_task_activity_source_rows(experiment: Any) -> pd.DataFrame:
    if experiment.source_results_csv is None:
        raise ValueError("`source_results_csv` is required for task_activity runs.")

    source_path = experiment.source_results_csv
    if not source_path.exists():
        raise FileNotFoundError(
            f"Task-activity source results file not found: {source_path}"
        )

    source_df = pd.read_csv(source_path)
    if "transcript_id" not in source_df.columns:
        raise ValueError("Task-activity source results must include `transcript_id`.")
    if "transcript" not in source_df.columns:
        raise ValueError(
            "Task-activity source results must include `transcript`. "
            "Re-run collection with `--include-transcript`."
        )

    rows: list[dict[str, Any]] = []
    mechanisms = experiment.mechanisms or MECHANISM_KEYS
    levels = set(experiment.levels or ["potential", "clear"])
    for _, row in source_df.iterrows():
        transcript_id = _string_value(row["transcript_id"])
        transcript = _string_value(row.get("transcript"))
        for mechanism in mechanisms:
            level = _string_value(row.get(level_column(mechanism)))
            if level not in levels:
                continue

            mechanism_label = (
                mechanism.replace("_", " ").replace(" opacity", "").title() + " opacity"
            )

            rows.append(
                {
                    "custom_id": f"{transcript_id}__{mechanism}",
                    "transcript_id": transcript_id,
                    "transcript": transcript,
                    "mechanism": mechanism,
                    "mechanism_label": mechanism_label,
                    "mechanism_context": TASK_ACTIVITY_MECHANISM_CONTEXTS[mechanism],
                    "level": level,
                    "form": _string_value(row.get(form_column(mechanism))),
                    "rationale": _string_value(row.get(rationale_column(mechanism))),
                    "evidence": _string_value(row.get(evidence_column(mechanism))),
                }
            )

    return pd.DataFrame(rows)


def _opacity_base_row(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    body = response.get("body") or {}
    usage = body.get("usage") or {}
    row: dict[str, Any] = {
        "transcript_id": record.get("custom_id", ""),
        "response_status_code": response.get("status_code"),
        "response_id": body.get("id", ""),
        "model": body.get("model", ""),
        "summary": "",
        "parse_error": "",
        "raw_output_text": "",
        "error": json.dumps(record.get("error")) if record.get("error") else "",
        "response_error": json.dumps(body.get("error")) if body.get("error") else "",
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": (
            usage.get("output_tokens_details") or {}
        ).get("reasoning_tokens"),
    }

    for mechanism_key in MECHANISM_KEYS:
        row[level_column(mechanism_key)] = ""
        row[form_column(mechanism_key)] = ""
        row[rationale_column(mechanism_key)] = ""
        row[evidence_column(mechanism_key)] = ""

    return row


def flatten_opacity_output_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one opacity-coding output row into CSV-ready columns."""
    row = _opacity_base_row(record)
    output_text = _extract_output_text(record)
    if not output_text:
        row["parse_error"] = "No assistant output_text found in response body."
        return row

    row["raw_output_text"] = output_text

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        row["parse_error"] = f"Assistant output was not valid JSON: {exc}"
        return row

    summary = payload.get("summary")
    if isinstance(summary, str):
        row["summary"] = summary

    mechanisms = payload.get("mechanisms")
    if not isinstance(mechanisms, dict):
        row["parse_error"] = "Parsed output JSON is missing a `mechanisms` object."
        return row

    for mechanism_key in MECHANISM_KEYS:
        mechanism = mechanisms.get(mechanism_key)
        if not isinstance(mechanism, dict):
            row["parse_error"] = (
                f"Parsed output JSON is missing mechanism `{mechanism_key}`."
            )
            continue

        level = _coerce_level(mechanism)
        row[level_column(mechanism_key)] = level
        row[form_column(mechanism_key)] = _coerce_form(mechanism, level)
        rationale = mechanism.get("rationale")
        row[rationale_column(mechanism_key)] = (
            rationale if isinstance(rationale, str) else ""
        )
        row[evidence_column(mechanism_key)] = _join_evidence(mechanism.get("evidence"))

    return row


def _export_opacity_coding_results_csv(
    *,
    run_dir: Path,
    include_transcript: bool,
    output_path: str | Path | None = None,
) -> Path:
    input_path = run_dir / "output.jsonl"
    input_jsonl_path = run_dir / "input.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Output file not found: {input_path}")

    resolved_output_path = (
        Path(output_path) if output_path is not None else run_dir / "results.csv"
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_map = (
        _load_transcript_map(input_jsonl_path) if include_transcript else {}
    )

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                bad_row = {
                    "transcript_id": "",
                    "response_status_code": "",
                    "response_id": "",
                    "model": "",
                    "summary": "",
                    "parse_error": f"Invalid JSONL row at line {line_number}: {exc}",
                    "raw_output_text": stripped,
                    "error": "",
                    "response_error": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                    "reasoning_tokens": "",
                    **{
                        column: ""
                        for mechanism_key in MECHANISM_KEYS
                        for column in (
                            level_column(mechanism_key),
                            form_column(mechanism_key),
                            rationale_column(mechanism_key),
                            evidence_column(mechanism_key),
                        )
                    },
                }
                if include_transcript:
                    bad_row["transcript"] = ""
                rows.append(bad_row)
                continue

            row = flatten_opacity_output_record(record)
            if include_transcript:
                row["transcript"] = transcript_map.get(row["transcript_id"], "")
            rows.append(row)

    fieldnames = ["transcript_id"]
    if include_transcript:
        fieldnames.append("transcript")
    fieldnames.extend(
        column for column in BASE_RESULTS_COLUMNS if column != "transcript_id"
    )
    for mechanism_key in MECHANISM_KEYS:
        fieldnames.extend(
            [
                level_column(mechanism_key),
                form_column(mechanism_key),
                rationale_column(mechanism_key),
                evidence_column(mechanism_key),
            ]
        )

    with resolved_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return resolved_output_path.resolve()


def _task_activity_base_row(
    record: dict[str, Any],
    source_row: dict[str, Any],
) -> dict[str, Any]:
    response = record.get("response") or {}
    body = response.get("body") or {}
    usage = body.get("usage") or {}
    return {
        "transcript_id": source_row.get("transcript_id", ""),
        "mechanism": source_row.get("mechanism", ""),
        "level": source_row.get("level", ""),
        "form": source_row.get("form", ""),
        "rationale": source_row.get("rationale", ""),
        "evidence": source_row.get("evidence", ""),
        "task_activities": "",
        "response_status_code": response.get("status_code"),
        "response_id": body.get("id", ""),
        "model": body.get("model", ""),
        "parse_error": "",
        "raw_output_text": "",
        "error": json.dumps(record.get("error")) if record.get("error") else "",
        "response_error": json.dumps(body.get("error")) if body.get("error") else "",
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": (
            usage.get("output_tokens_details") or {}
        ).get("reasoning_tokens"),
    }


def _parse_task_activities_output(output_text: str) -> list[str]:
    cleaned = output_text.strip()
    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Backward compatibility with earlier single-string outputs.
        return [cleaned]

    if isinstance(payload, str):
        normalized = payload.strip()
        return [normalized] if normalized else []

    if not isinstance(payload, list):
        raise ValueError("Assistant output must be a JSON array of strings.")

    activities: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError("Assistant output array must contain only strings.")
        normalized = item.strip()
        if normalized:
            activities.append(normalized)

    return activities


def flatten_task_activity_output_record(
    record: dict[str, Any],
    *,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one task-activity labeling output row."""
    row = _task_activity_base_row(record, source_row)
    output_text = _extract_output_text(record)
    if not output_text:
        row["parse_error"] = "No assistant output_text found in response body."
        return row

    cleaned = output_text.strip()
    row["raw_output_text"] = cleaned
    if not cleaned:
        row["parse_error"] = "Assistant output_text was empty."
        return row

    try:
        activities = _parse_task_activities_output(cleaned)
    except ValueError as exc:
        row["parse_error"] = str(exc)
        return row

    if not activities:
        row["parse_error"] = "Assistant output did not contain any activity strings."
        return row

    row["task_activities"] = json.dumps(activities, ensure_ascii=False)
    return row


def _task_activity_source_map(experiment: Any) -> dict[str, dict[str, Any]]:
    source_df = _load_task_activity_source_rows(experiment)
    return {
        _string_value(row["custom_id"]): {
            "transcript_id": _string_value(row["transcript_id"]),
            "transcript": _string_value(row.get("transcript")),
            "mechanism": _string_value(row["mechanism"]),
            "level": _string_value(row["level"]),
            "form": _string_value(row["form"]),
            "rationale": _string_value(row["rationale"]),
            "evidence": _string_value(row["evidence"]),
        }
        for _, row in source_df.iterrows()
    }


def _export_task_activity_results_csv(
    *,
    experiment: Any,
    run_dir: Path,
    include_transcript: bool,
    output_path: str | Path | None = None,
) -> Path:
    input_path = run_dir / "output.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Output file not found: {input_path}")

    resolved_output_path = (
        Path(output_path) if output_path is not None else run_dir / "results.csv"
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    source_map = _task_activity_source_map(experiment)

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                bad_row = {
                    key: ""
                    for key in TASK_ACTIVITY_RESULTS_COLUMNS
                    if key != "parse_error"
                }
                bad_row["parse_error"] = (
                    f"Invalid JSONL row at line {line_number}: {exc}"
                )
                if include_transcript:
                    bad_row["transcript"] = ""
                rows.append(bad_row)
                continue

            custom_id = _string_value(record.get("custom_id"))
            source_row = source_map.get(custom_id)
            if source_row is None:
                source_row = {
                    "transcript_id": "",
                    "transcript": "",
                    "mechanism": "",
                    "level": "",
                    "form": "",
                    "rationale": "",
                    "evidence": "",
                }
                row = _task_activity_base_row(record, source_row)
                row["parse_error"] = (
                    f"No source metadata found for custom_id `{custom_id}`."
                )
            else:
                row = flatten_task_activity_output_record(record, source_row=source_row)

            if include_transcript:
                row["transcript"] = source_row.get("transcript", "")
            rows.append(row)

    fieldnames = list(TASK_ACTIVITY_RESULTS_COLUMNS)
    if include_transcript:
        fieldnames.insert(1, "transcript")

    with resolved_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return resolved_output_path.resolve()
