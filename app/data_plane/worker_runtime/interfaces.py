from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from app.schemas.model import ModelMetadata


class LoadedRuntimeModel(Protocol):
    name: str
    version: str
    metadata: ModelMetadata
    artifact_path: Path

    @property
    def pid(self) -> int:
        """Operating-system process id of the loaded worker."""


class RuntimeSupervisor(Protocol):
    """Supervises loaded model workers and routes tensor inference to them."""

    cpu_budget: int

    async def load(
        self,
        model_name: str,
        version: str,
        artifact_path: Path,
        metadata: ModelMetadata,
    ) -> dict[str, Any]:
        """Load a model version and replace the previous active worker."""

    async def create_session(
        self,
        model_name: str,
        version: str,
        artifact_path: Path,
        metadata: ModelMetadata,
    ) -> LoadedRuntimeModel:
        """Create a worker session without making it active."""

    async def replace_loaded(
        self, model_name: str, loaded: LoadedRuntimeModel
    ) -> dict[str, Any]:
        """Make a prepared worker active and unload the previous one."""

    async def unload(self, model_name: str) -> dict[str, Any]:
        """Unload an active model worker."""

    async def shutdown_all(self) -> dict[str, Any]:
        """Unload every active model worker."""

    async def get_loaded(self, model_name: str) -> LoadedRuntimeModel | None:
        """Return a loaded model, restarting crashed workers when possible."""

    def loaded_models(self) -> dict[str, str]:
        """Return currently loaded model versions."""

    def process_metrics(self) -> dict[str, dict[str, float | int | str]]:
        """Return process metrics for loaded workers."""

    def cpu_requested_threads(self) -> int:
        """Return total requested inference threads across loaded workers."""

    def worker_crashes(self, model_name: str, version: str | None = None) -> int:
        """Return worker crash count for a model or model version."""

    def worker_restart_failures(
        self, model_name: str, version: str | None = None
    ) -> int:
        """Return failed worker restart count for a model or model version."""

    async def infer_tensors(
        self,
        model_name: str,
        inputs: list[dict[str, Any]],
        payload: bytes,
        output_names: list[str],
    ) -> tuple[list[dict[str, Any]], bytes, dict[str, float]]:
        """Run tensor inference against an active model worker."""

    async def infer_loaded(
        self,
        loaded: LoadedRuntimeModel,
        inputs: list[dict[str, Any]],
        payload: bytes,
        output_names: list[str],
    ) -> tuple[list[dict[str, Any]], bytes, dict[str, float]]:
        """Run tensor inference against a prepared worker."""

    async def discard_session(self, loaded: LoadedRuntimeModel) -> dict[str, Any]:
        """Terminate a prepared worker that was not made active."""
