from pathlib import Path
from types import SimpleNamespace

import pytest

from interviewer.batch.models import ExperimentConfig, RunContext
from interviewer.batch.providers.openai_provider import OpenAIProvider


class _FakeTextResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeFiles:
    def __init__(self):
        self.created = []

    def create(self, file, purpose: str):
        _ = file.read()
        self.created.append(purpose)
        return SimpleNamespace(id="file-input-1")

    def content(self, file_id: str):
        return _FakeTextResponse(f"content for {file_id}")


class _FakeBatches:
    def __init__(self):
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(
            id="batch-1",
            status="validating",
            model_dump=lambda: {"id": "batch-1", "status": "validating"},
        )

    def retrieve(self, batch_id: str):
        return SimpleNamespace(
            id=batch_id,
            status="completed",
            output_file_id="file-out-1",
            error_file_id="file-err-1",
            model_dump=lambda: {
                "id": batch_id,
                "status": "completed",
                "output_file_id": "file-out-1",
                "error_file_id": "file-err-1",
            },
        )


class _FakeClient:
    def __init__(self):
        self.files = _FakeFiles()
        self.batches = _FakeBatches()


def test_openai_provider_submit_and_collect(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text('{"x":1}\n', encoding="utf-8")

    experiment = ExperimentConfig(
        name="exp1",
        provider="openai",
        model="gpt-5.2-mini",
        prompt_file=tmp_path / "prompt.txt",
        metadata={"owner": "qa"},
    )
    run_ctx = RunContext(experiment=experiment, input_jsonl_path=input_jsonl)

    client = _FakeClient()
    provider = OpenAIProvider(client=client)

    submission = provider.submit_batch(run_ctx)
    assert submission.batch_id == "batch-1"
    assert submission.input_file_id == "file-input-1"
    assert client.batches.create_calls[0]["endpoint"] == "/v1/responses"
    assert client.batches.create_calls[0]["completion_window"] == "24h"

    status = provider.get_batch_status("batch-1")
    assert status.status == "completed"
    assert status.output_file_id == "file-out-1"

    downloaded = provider.download_file("file-out-1")
    assert downloaded == "content for file-out-1"


def test_openai_provider_requires_api_key_without_injected_client(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider(client=None)
