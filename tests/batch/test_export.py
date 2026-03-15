import csv
import json
from pathlib import Path

from interviewer.batch.export import export_results_csv, flatten_output_record
from interviewer.batch import orchestrator


def _sample_record(custom_id: str = "work_0000") -> dict:
    payload = {
        "summary": "A short analytic summary.",
        "mechanisms": {
            "voice_opacity": {
                "level": "none",
                "form": "none",
                "evidence": [],
                "rationale": "No evidence of voice filtering.",
            },
            "vulnerability_opacity": {
                "level": "potential",
                "form": "production",
                "evidence": ["Speaker uses AI when stuck."],
                "rationale": "AI masks uncertainty.",
            },
            "provenance_opacity": {
                "level": "none",
                "form": "none",
                "evidence": [],
                "rationale": "No provenance issue.",
            },
            "attention_opacity": {
                "level": "none",
                "form": "none",
                "evidence": [],
                "rationale": "No attention substitution.",
            },
            "investment_opacity": {
                "level": "none",
                "form": "none",
                "evidence": [],
                "rationale": "No effort-obscuring dynamic.",
            },
        },
    }

    return {
        "custom_id": custom_id,
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
                                "text": json.dumps(payload),
                            }
                        ],
                    },
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                    "output_tokens_details": {"reasoning_tokens": 20},
                },
                "error": None,
            },
        },
        "error": None,
    }


def test_flatten_output_record_uses_current_prompt_shape() -> None:
    row = flatten_output_record(_sample_record())

    assert row["transcript_id"] == "work_0000"
    assert row["summary"] == "A short analytic summary."
    assert row["vulnerability_opacity_level"] == "potential"
    assert row["vulnerability_opacity_form"] == "production"
    assert row["vulnerability_opacity_evidence"] == "Speaker uses AI when stuck."
    assert row["voice_opacity_level"] == "none"
    assert row["voice_opacity_form"] == "none"
    assert row["parse_error"] == ""


def test_export_results_csv_for_test_run(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    experiment_path = tmp_path / "exp.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "name: export-exp",
                "provider: openai",
                "model: gpt-5.4",
                f"prompt_file: {prompt_path}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator, "RUNS_ROOT", tmp_path / "outputs" / "runs")
    run_dir = orchestrator._run_dir_for("export-exp", test_mode=True)
    run_dir.mkdir(parents=True)
    output_jsonl = run_dir / "output.jsonl"
    output_jsonl.write_text(
        "\n".join(
            [
                json.dumps(_sample_record("work_0000")),
                json.dumps(_sample_record("work_0001")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = export_results_csv(experiment_path, test_mode=True)
    assert csv_path == (run_dir / "results.csv").resolve()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["transcript_id"] == "work_0000"
    assert rows[0]["summary"] == "A short analytic summary."
    assert rows[0]["vulnerability_opacity_level"] == "potential"
    assert rows[0]["vulnerability_opacity_form"] == "production"
    assert rows[0]["response_status_code"] == "200"


def test_export_results_csv_handles_invalid_assistant_json(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    experiment_path = tmp_path / "exp.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "name: bad-export-exp",
                "provider: openai",
                "model: gpt-5.4",
                f"prompt_file: {prompt_path}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator, "RUNS_ROOT", tmp_path / "outputs" / "runs")
    run_dir = orchestrator._run_dir_for("bad-export-exp")
    run_dir.mkdir(parents=True)
    bad_record = _sample_record()
    bad_record["response"]["body"]["output"][1]["content"][0]["text"] = "{not valid json"
    (run_dir / "output.jsonl").write_text(json.dumps(bad_record) + "\n", encoding="utf-8")

    csv_path = export_results_csv(experiment_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert "not valid JSON" in rows[0]["parse_error"]


def test_flatten_output_record_supports_legacy_present_bool() -> None:
    legacy_record = _sample_record()
    mechanisms = json.loads(
        legacy_record["response"]["body"]["output"][1]["content"][0]["text"]
    )["mechanisms"]
    for mechanism in mechanisms.values():
        mechanism.pop("form")
        mechanism["present"] = mechanism.pop("level") != "none"
    legacy_record["response"]["body"]["output"][1]["content"][0]["text"] = json.dumps(
        {
            "summary": "Legacy summary.",
            "mechanisms": mechanisms,
        }
    )

    row = flatten_output_record(legacy_record)

    assert row["summary"] == "Legacy summary."
    assert row["vulnerability_opacity_level"] == "clear"
    assert row["vulnerability_opacity_form"] == ""
    assert row["voice_opacity_level"] == "none"
    assert row["voice_opacity_form"] == "none"


def test_flatten_output_record_supports_level_only_schema() -> None:
    level_only_record = _sample_record()
    payload = json.loads(level_only_record["response"]["body"]["output"][1]["content"][0]["text"])
    for mechanism in payload["mechanisms"].values():
        mechanism.pop("form")
    level_only_record["response"]["body"]["output"][1]["content"][0]["text"] = json.dumps(payload)

    row = flatten_output_record(level_only_record)

    assert row["vulnerability_opacity_level"] == "potential"
    assert row["vulnerability_opacity_form"] == ""
    assert row["voice_opacity_level"] == "none"
    assert row["voice_opacity_form"] == "none"
