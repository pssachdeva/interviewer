"""Core types for batch experiment submission and collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled", "canceled"}


@dataclass(slots=True)
class ExperimentConfig:
    """Experiment configuration loaded from YAML."""

    name: str
    provider: str
    model: str
    prompt_file: Path
    reasoning_effort: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


@dataclass(slots=True)
class RunContext:
    """Submission context provided to provider clients."""

    experiment: ExperimentConfig
    input_jsonl_path: Path
    completion_window: str = "24h"


@dataclass(slots=True)
class ProviderSubmission:
    """Provider response returned right after batch creation."""

    batch_id: str
    input_file_id: str
    status: str
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderBatchStatus:
    """Provider status snapshot for an existing batch."""

    batch_id: str
    status: str
    output_file_id: str | None = None
    error_file_id: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunManifest:
    """Persistent local record for an experiment run."""

    experiment_name: str
    experiment_path: str
    provider: str
    model: str
    prompt_file: str
    prompt_sha256: str
    experiment_sha256: str
    created_at_utc: str
    run_dir: str
    input_jsonl_path: str
    test_mode: bool = False
    transcript_count: int | None = None
    batch_id: str | None = None
    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    status: str = "created"
    collected_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunManifest":
        return cls(**payload)


@dataclass(slots=True)
class CollectResult:
    """Result returned after a collect operation."""

    manifest: RunManifest
    status: ProviderBatchStatus
    status_path: Path
    output_path: Path | None = None
    error_path: Path | None = None
    csv_path: Path | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.status in TERMINAL_BATCH_STATUSES
