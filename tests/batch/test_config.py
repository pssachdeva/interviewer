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


@pytest.mark.parametrize(
    "content,error_substring",
    [
        ("provider: openai\nprompt_file: foo.txt\n", "`name`"),
        ("name: x\nprompt_file: foo.txt\n", "`provider`"),
        ("name: x\nprovider: anthropic\nprompt_file: foo.txt\n", "Only `openai`"),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\nreasoning_effort: max\n",
            "`reasoning_effort` must be one of",
        ),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\nreasoning_effort: high\nrequest:\n  reasoning:\n    effort: low\n",
            "Use either top-level `reasoning_effort` or `request.reasoning`",
        ),
        (
            "name: x\nprovider: openai\nprompt_file: foo.txt\nextra: 1\n",
            "Unknown experiment field",
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
