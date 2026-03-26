from pathlib import Path

import pytest

from interviewer.batch.config import load_experiment_config


def test_load_experiment_config_success(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    prompt_file = prompts / "prompt.txt"
    prompt_file.write_text("system prompt", encoding="utf-8")

    experiment = tmp_path / "exp.yaml"
    experiment.write_text(
        "\n".join(
            [
                "name: test-exp",
                "provider: openai",
                "model: gpt-5.2-mini",
                "prompt_file: prompts/prompt.txt",
                "reasoning_effort: high",
                "request:",
                "  temperature: 0",
                "metadata:",
                "  owner: qa",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)

    assert cfg.name == "test-exp"
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-5.2-mini"
    assert cfg.prompt_file == prompt_file.resolve()
    assert cfg.reasoning_effort == "high"
    assert cfg.request == {"temperature": 0}
    assert cfg.metadata == {"owner": "qa"}


def test_load_experiment_config_defaults_model(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("p", encoding="utf-8")

    experiment = tmp_path / "exp.yaml"
    experiment.write_text(
        "\n".join(
            [
                "name: test-exp",
                "provider: openai",
                f"prompt_file: {prompt_file}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)
    assert cfg.model == "gpt-5.2-mini"


def test_load_experiment_config_defaults_task_type(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("p", encoding="utf-8")

    experiment = tmp_path / "exp.yaml"
    experiment.write_text(
        "\n".join(
            [
                "name: test-exp",
                "provider: openai",
                f"prompt_file: {prompt_file}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)
    assert cfg.task_type == "opacity_coding"


def test_load_experiment_config_task_activity_fields(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt", encoding="utf-8")
    results_file = tmp_path / "results.csv"
    results_file.write_text(
        "transcript_id,transcript\nwork_0000,text\n",
        encoding="utf-8",
    )

    experiment = tmp_path / "task-activity.yaml"
    experiment.write_text(
        "\n".join(
            [
                "name: task-activity",
                "provider: openai",
                "task_type: task_activity",
                f"prompt_file: {prompt_file}",
                f"source_results_csv: {results_file}",
                "mechanisms:",
                "  - voice_opacity",
                "levels:",
                "  - potential",
                "  - clear",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(experiment)
    assert cfg.task_type == "task_activity"
    assert cfg.source_results_csv == results_file.resolve()
    assert cfg.mechanisms == ["voice_opacity"]
    assert cfg.levels == ["potential", "clear"]


@pytest.mark.parametrize(
    "content,error_substring",
    [
        ("provider: openai\nprompt_file: foo.txt\n", "`name`"),
        ("name: x\nprompt_file: foo.txt\n", "`provider`"),
        ("name: x\nprovider: anthropic\nprompt_file: foo.txt\n", "Only `openai`"),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\n"
            "reasoning_effort: max\n",
            "`reasoning_effort` must be one of",
        ),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\n"
            "reasoning_effort: high\nrequest:\n  reasoning:\n    effort: low\n",
            "Use either top-level `reasoning_effort` or `request.reasoning`",
        ),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\nextra: 1\n",
            "Unknown experiment field",
        ),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\n"
            "task_type: task_activity\n",
            "`source_results_csv` is required",
        ),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\ntask_type: unknown\n",
            "`task_type` must be one of",
        ),
    ],
)
def test_load_experiment_config_validation_errors(
    tmp_path: Path, content: str, error_substring: str
) -> None:
    experiment = tmp_path / "exp.yaml"
    experiment.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=error_substring):
        load_experiment_config(experiment)
