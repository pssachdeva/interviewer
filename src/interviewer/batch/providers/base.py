"""Provider client interface for batch APIs."""

from __future__ import annotations

from typing import Protocol

from interviewer.batch.models import ProviderBatchStatus, ProviderSubmission, RunContext


class ProviderClient(Protocol):
    """Common interface required by all batch provider implementations."""

    def submit_batch(self, run_ctx: RunContext) -> ProviderSubmission:
        """Upload input and submit a batch job."""

    def get_batch_status(self, batch_id: str) -> ProviderBatchStatus:
        """Fetch provider status for a batch id."""

    def download_file(self, file_id: str) -> str:
        """Download provider file content as text."""
