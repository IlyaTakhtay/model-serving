from __future__ import annotations

from typing import Any

from app.control_plane.model_management.catalog import ModelCatalog
from app.control_plane.model_management.lifecycle import ModelLifecycle
from app.control_plane.model_management.uploader import ModelUploader
from app.control_plane.rollback.policy_manager import AutoRollbackPolicyManager
from app.schemas.upload import ModelUploadChunkRequest, ModelUploadRequest


class ModelControlService:
    def __init__(
        self,
        catalog: ModelCatalog,
        lifecycle: ModelLifecycle,
        uploader: ModelUploader,
        rollback_policies: AutoRollbackPolicyManager,
    ) -> None:
        self._catalog = catalog
        self._lifecycle = lifecycle
        self._uploader = uploader
        self._rollback_policies = rollback_policies

    def list_models(self) -> dict[str, Any]:
        return self._catalog.list_models()

    def describe_model(self, model_name: str) -> dict[str, Any]:
        return self._catalog.describe_model(model_name)

    async def upload(
        self, model_name: str, version: str, data: ModelUploadRequest
    ) -> dict[str, Any]:
        return await self._uploader.upload(model_name, version, data)

    async def upload_chunk(
        self, model_name: str, version: str, data: ModelUploadChunkRequest
    ) -> dict[str, Any]:
        return await self._uploader.upload_chunk(model_name, version, data)

    async def activate(self, model_name: str, version: str) -> dict[str, Any]:
        return await self._lifecycle.activate(model_name, version)

    async def deactivate(self, model_name: str) -> dict[str, Any]:
        return await self._lifecycle.deactivate(model_name)

    async def rollback(self, model_name: str) -> dict[str, Any]:
        return await self._lifecycle.rollback(model_name)

    def loaded_models(self) -> dict[str, str]:
        return self._catalog.loaded_models()

    def get_active_rollback_policy(self, model_name: str) -> dict[str, Any]:
        return self._rollback_policies.get_active_policy(model_name)

    async def set_active_rollback_policy(
        self, model_name: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._rollback_policies.set_active_policy(model_name, data)

    async def disable_active_rollback_policy(self, model_name: str) -> dict[str, Any]:
        return await self._rollback_policies.disable_active_policy(model_name)

    def get_version_rollback_policy(
        self, model_name: str, version: str
    ) -> dict[str, Any]:
        return self._rollback_policies.get_policy(model_name, version)

    async def set_version_rollback_policy(
        self, model_name: str, version: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._rollback_policies.set_policy(model_name, version, data)

    async def disable_version_rollback_policy(
        self, model_name: str, version: str
    ) -> dict[str, Any]:
        return await self._rollback_policies.disable_policy(model_name, version)
