from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.schemas.model import ModelMetadata


class ModelArtifactInspector(Protocol):
    """Inspects a stored artifact and returns serving metadata."""

    runtime: str

    def inspect(
        self, model_name: str, version: str, artifact: str, artifact_path: Path
    ) -> ModelMetadata:
        """Read artifact inputs, outputs, runtime and execution metadata."""
