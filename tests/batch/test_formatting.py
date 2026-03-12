from pathlib import Path

import pandas as pd
import pytest

from interviewer.batch.formatting import build_batch_line, iter_batch_lines, write_jsonl


def test_build_batch_line_shape() -> None:
    line = build_batch_line(
        transcript_id="work_0001",
        transcript_text="transcript text",
        system_prompt="system",
        model="gpt-5.2-mini",
        reasoning_effort="medium",
        request_overrides={"temperature": 0},
    )

    assert line["custom_id"] == "work_0001"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/responses"
    assert line["body"]["model"] == "gpt-5.2-mini"
    assert line["body"]["reasoning"] == {"effort": "medium"}
    assert line["body"]["temperature"] == 0
    assert line["body"]["input"][0]["role"] == "system"
    assert line["body"]["input"][0]["content"] == "system"
    assert line["body"]["input"][1]["role"] == "user"
    assert line["body"]["input"][1]["content"] == "transcript text"


def test_iter_batch_lines_and_write_jsonl(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"transcript_id": "work_0000", "text": "a"},
            {"transcript_id": "work_0001", "text": "b"},
        ]
    )

    rows = list(
        iter_batch_lines(
            df,
            system_prompt="prompt",
            model="gpt-5.2-mini",
            reasoning_effort="low",
        )
    )
    assert len(rows) == 2
    assert rows[0]["custom_id"] == "work_0000"
    assert rows[1]["custom_id"] == "work_0001"
    assert rows[0]["body"]["reasoning"] == {"effort": "low"}

    out = tmp_path / "input.jsonl"
    write_jsonl(out, rows)

    text = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2
    assert '"custom_id": "work_0000"' in text[0]


def test_iter_batch_lines_missing_columns_raises() -> None:
    df = pd.DataFrame([{"id": "x", "content": "y"}])
    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_batch_lines(df, system_prompt="prompt", model="gpt-5.2-mini"))
