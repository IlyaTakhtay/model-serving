from __future__ import annotations

from typing import Any

import msgspec

from app.common.exceptions import ModelNotFoundError
from app.control_plane.model_management.active_state import ActiveModelStateStore
from app.control_plane.model_management.registry import ModelRegistry
from app.data_plane.worker_runtime.runtime import WorkerRuntime


class ModelCatalog:
    def __init__(
        self,
        registry: ModelRegistry,
        active_state: ActiveModelStateStore,
        runtime: WorkerRuntime,
    ) -> None:
        self.registry = registry
        self.active_state = active_state
        self.runtime = runtime

    def list_models(self) -> dict[str, Any]:
        return {
            "models": [
                self.describe_model(name) for name in self.registry.list_models()
            ]
        }

    def loaded_models(self) -> dict[str, str]:
        return self.runtime.loaded_models()

    def describe_model(self, model_name: str) -> dict[str, Any]:
        if model_name not in self.registry.list_models():
            raise ModelNotFoundError(f"Model '{model_name}' is not registered")
        versions = self.registry.list_versions(model_name)
        active = self.active_state.get_active_version(model_name)
        loaded_version = self.runtime.loaded_models().get(model_name)
        metadata = (
            msgspec.to_builtins(self.registry.get_metadata(model_name, active))
            if active
            else None
        )
        return {
            "name": model_name,
            "versions": versions,
            "active_version": active,
            "previous_version": self.active_state.get_previous_version(model_name),
            "ready": loaded_version is not None,
            "loaded_version": loaded_version,
            "metadata": metadata,
        }
