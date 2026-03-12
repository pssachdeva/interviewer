"""OpenAI Batch API provider implementation."""

from __future__ import annotations

import os
from typing import Any

from interviewer.batch.models import ProviderBatchStatus, ProviderSubmission, RunContext
from interviewer.batch.providers.base import ProviderClient


BATCH_ENDPOINT = "/v1/responses"
DEFAULT_COMPLETION_WINDOW = "24h"


def _coerce_response_to_dict(obj: Any) -> dict[str, Any]:
    """Convert SDK objects into plain dictionaries for persistence."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return value

    raw = getattr(obj, "__dict__", None)
    if isinstance(raw, dict):
        return raw

    return {"value": str(obj)}


class OpenAIProvider(ProviderClient):
    """Provider adapter for OpenAI Batch API."""

    def __init__(self, client: Any | None = None, completion_window: str = DEFAULT_COMPLETION_WINDOW):
        self.completion_window = completion_window

        if client is not None:
            self.client = client
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required. Install dependencies with `uv pip install -e .`."
            ) from exc

        self.client = OpenAI(api_key=api_key)

    def submit_batch(self, run_ctx: RunContext) -> ProviderSubmission:
        with run_ctx.input_jsonl_path.open("rb") as f:
            input_file = self.client.files.create(file=f, purpose="batch")

        metadata = {"experiment_name": run_ctx.experiment.name}
        for key, value in run_ctx.experiment.metadata.items():
            metadata[str(key)] = str(value)

        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint=BATCH_ENDPOINT,
            completion_window=run_ctx.completion_window or self.completion_window,
            metadata=metadata,
        )

        return ProviderSubmission(
            batch_id=batch.id,
            input_file_id=input_file.id,
            status=getattr(batch, "status", "submitted"),
            raw_response=_coerce_response_to_dict(batch),
        )

    def get_batch_status(self, batch_id: str) -> ProviderBatchStatus:
        batch = self.client.batches.retrieve(batch_id)
        return ProviderBatchStatus(
            batch_id=batch.id,
            status=getattr(batch, "status", "unknown"),
            output_file_id=getattr(batch, "output_file_id", None),
            error_file_id=getattr(batch, "error_file_id", None),
            raw_response=_coerce_response_to_dict(batch),
        )

    def download_file(self, file_id: str) -> str:
        content = self.client.files.content(file_id)

        text = getattr(content, "text", None)
        if isinstance(text, str):
            return text

        read = getattr(content, "read", None)
        if callable(read):
            data = read()
            if isinstance(data, bytes):
                return data.decode("utf-8")
            return str(data)

        if isinstance(content, bytes):
            return content.decode("utf-8")

        return str(content)
