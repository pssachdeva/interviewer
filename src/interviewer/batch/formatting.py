"""Formatting helpers for OpenAI batch JSONL inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from interviewer.batch.providers.openai_provider import BATCH_ENDPOINT


def build_request_body(
    model: str,
    system_prompt: str,
    user_message: str,
    reasoning_effort: str | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single OpenAI Responses request body for batch submission."""
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}

    if request_overrides:
        body.update(request_overrides)

    return body


def build_batch_line(
    transcript_id: str,
    transcript_text: str,
    system_prompt: str,
    model: str,
    reasoning_effort: str | None = None,
    request_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
    """Build one JSONL line for OpenAI Batch API."""
    return build_batch_line_from_messages(
        custom_id=transcript_id,
        system_prompt=system_prompt,
        user_message=transcript_text,
        model=model,
        reasoning_effort=reasoning_effort,
        request_overrides=request_overrides,
    )


def build_batch_line_from_messages(
    custom_id: str,
    system_prompt: str,
    user_message: str,
    model: str,
    reasoning_effort: str | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSONL line from arbitrary system and user messages."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": build_request_body(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            reasoning_effort=reasoning_effort,
            request_overrides=request_overrides,
        ),
    }


def iter_batch_lines(
    interviews_df: pd.DataFrame,
    system_prompt: str,
    model: str,
    reasoning_effort: str | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield batch request lines for all interviews in a DataFrame."""
    required_columns = {"transcript_id", "text"}
    missing = required_columns - set(interviews_df.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Interview DataFrame missing required columns: {joined}")

    for _, row in interviews_df.iterrows():
        transcript_id = str(row["transcript_id"])
        transcript_text = "" if row["text"] is None else str(row["text"])

        yield build_batch_line(
            transcript_id=transcript_id,
            transcript_text=transcript_text,
            system_prompt=system_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            request_overrides=request_overrides,
        )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dict rows to JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
