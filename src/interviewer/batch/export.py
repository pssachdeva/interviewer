"""Export raw batch outputs into a flat CSV for downstream analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from interviewer.batch.config import load_experiment_config
from interviewer.batch.orchestrator import _run_dir_for
from interviewer.opacity_columns import (
    BASE_RESULTS_COLUMNS,
    FORMS,
    LEVELS,
    MECHANISM_KEYS,
    evidence_column,
    form_column,
    level_column,
    rationale_column,
)


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
    if isinstance(level, str) and level in LEVELS:
        return level

    # Backward compatibility for earlier outputs that used present=true/false.
    present = mechanism.get("present")
    if present is True:
        return "clear"
    if present is False:
        return "none"

    return ""


def _coerce_form(mechanism: dict[str, Any], level: str) -> str:
    form = mechanism.get("form")
    if isinstance(form, str) and form in FORMS:
        return form

    if level == "none":
        return "none"

    # Backward compatibility for older outputs that did not include `form`.
    return ""


def _base_row(record: dict[str, Any]) -> dict[str, Any]:
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
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
    }

    for mechanism_key in MECHANISM_KEYS:
        row[level_column(mechanism_key)] = ""
        row[form_column(mechanism_key)] = ""
        row[rationale_column(mechanism_key)] = ""
        row[evidence_column(mechanism_key)] = ""

    return row


def flatten_output_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one OpenAI batch output row into CSV-ready columns."""
    row = _base_row(record)
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
        row[rationale_column(mechanism_key)] = rationale if isinstance(rationale, str) else ""
        row[evidence_column(mechanism_key)] = _join_evidence(mechanism.get("evidence"))

    return row


def export_results_csv(
    experiment_path: str | Path,
    test_mode: bool = False,
    include_transcript: bool = False,
    output_path: str | Path | None = None,
) -> Path:
    """Export batch JSONL outputs for an experiment into a flat CSV file."""
    experiment = load_experiment_config(experiment_path)
    run_dir = _run_dir_for(experiment.name, test_mode=test_mode)
    input_path = run_dir / "output.jsonl"
    input_jsonl_path = run_dir / "input.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Output file not found: {input_path}")

    resolved_output_path = Path(output_path) if output_path is not None else run_dir / "results.csv"
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_map = _load_transcript_map(input_jsonl_path) if include_transcript else {}

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

            row = flatten_output_record(record)
            if include_transcript:
                row["transcript"] = transcript_map.get(row["transcript_id"], "")
            rows.append(row)

    fieldnames = ["transcript_id"]
    if include_transcript:
        fieldnames.append("transcript")
    fieldnames.extend(column for column in BASE_RESULTS_COLUMNS if column != "transcript_id")
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
