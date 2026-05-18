from __future__ import annotations

from typing import Any

from app.common.exceptions import ModelNotReadyError
from app.control_plane.model_management.active_state import ActiveModelStateStore
from app.control_plane.model_management.registry import ModelRegistry
from app.domain.rollback import RollbackPolicy
from app.observability.recorder import ObservabilityRecorder


class AutoRollbackPolicyManager:
    def __init__(
        self,
        registry: ModelRegistry,
        active_state: ActiveModelStateStore,
        recorder: ObservabilityRecorder,
    ) -> None:
        self.registry = registry
        self.active_state = active_state
        self.recorder = recorder

    def get_active_policy(self, model_name: str) -> dict[str, Any]:
        version = self._active_version(model_name)
        return self.get_policy(model_name, version)

    async def set_active_policy(
        self, model_name: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        version = self._active_version(model_name)
        return await self.set_policy(model_name, version, data)

    async def disable_active_policy(self, model_name: str) -> dict[str, Any]:
        version = self._active_version(model_name)
        return await self.disable_policy(model_name, version)

    def get_policy(self, model_name: str, version: str) -> dict[str, Any]:
        metadata = self.registry.get_metadata(model_name, version)
        policy = RollbackPolicy.from_dict(metadata.rollback_policy)
        return {
            "model_name": model_name,
            "model_version": version,
            "rollback_policy": policy.to_dict(),
        }

    async def set_policy(
        self, model_name: str, version: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        policy = RollbackPolicy.from_dict(data)
        metadata = self.registry.get_metadata(model_name, version)
        metadata.rollback_policy = policy.to_dict()
        self.registry.save_metadata(model_name, version, metadata)
        self.active_state.clear_auto_rollback_block(model_name)
        await self.recorder.record(
            "ROLLBACK_POLICY_UPDATED",
            model=model_name,
            version=version,
            enabled=policy.enabled,
            policy=metadata.rollback_policy,
        )
        return {
            "model_name": model_name,
            "model_version": version,
            "rollback_policy": metadata.rollback_policy,
        }

    async def disable_policy(self, model_name: str, version: str) -> dict[str, Any]:
        metadata = self.registry.get_metadata(model_name, version)
        current = RollbackPolicy.from_dict(metadata.rollback_policy)
        policy = RollbackPolicy(
            enabled=False,
            window_size=current.window_size,
            max_error_rate=current.max_error_rate,
            max_compute_infer_p95_ms=current.max_compute_infer_p95_ms,
            max_worker_crashes=current.max_worker_crashes,
            max_memory_mb=current.max_memory_mb,
            max_cpu_usage_percent_window=current.max_cpu_usage_percent_window,
        )
        metadata.rollback_policy = policy.to_dict()
        self.registry.save_metadata(model_name, version, metadata)
        self.active_state.clear_auto_rollback_block(model_name)
        await self.recorder.record(
            "ROLLBACK_POLICY_DISABLED", model=model_name, version=version
        )
        return {
            "model_name": model_name,
            "model_version": version,
            "rollback_policy": metadata.rollback_policy,
        }

    def _active_version(self, model_name: str) -> str:
        version = self.active_state.get_active_version(model_name)
        if not version:
            raise ModelNotReadyError(f"Model '{model_name}' has no active version")
        return version
