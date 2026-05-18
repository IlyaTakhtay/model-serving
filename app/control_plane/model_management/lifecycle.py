from __future__ import annotations

from typing import Any

import numpy as np

from app.common.exceptions import (
    ModelHealthCheckError,
    ServingError,
    VersionNotFoundError,
)
from app.common.tensor_datatypes import DATATYPE_TO_DTYPE
from app.control_plane.model_management.active_state import ActiveModelStateStore
from app.control_plane.model_management.registry import ModelRegistry
from app.data_plane.worker_runtime.runtime import LoadedModel, WorkerRuntime
from app.observability.recorder import ObservabilityRecorder
from app.schemas.model import ModelMetadata


class ModelLifecycle:
    def __init__(
        self,
        registry: ModelRegistry,
        active_state: ActiveModelStateStore,
        runtime: WorkerRuntime,
        recorder: ObservabilityRecorder,
    ) -> None:
        self.registry = registry
        self.active_state = active_state
        self.runtime = runtime
        self.recorder = recorder

    async def load_active_models(self) -> None:
        for model_name, state in self.active_state.read().items():
            version = state.get("active")
            if not version:
                continue
            try:
                await self._load(model_name, version)
                await self.recorder.record(
                    "MODEL_LOADED", model=model_name, version=version
                )
            except ServingError as exc:
                await self.recorder.record(
                    "MODEL_LOAD_FAILED",
                    model=model_name,
                    version=version,
                    error=exc.message,
                )

    async def activate(self, model_name: str, version: str) -> dict[str, Any]:
        previous = self.active_state.get_active_version(model_name)
        metadata = self.registry.get_metadata(model_name, version)
        artifact_path = self.registry.get_artifact_path(model_name, version)
        loaded = await self.runtime.create_session(
            model_name, version, artifact_path, metadata
        )
        try:
            healthcheck = await self._run_healthcheck(loaded, metadata)
        except Exception:
            await self.runtime.discard_session(loaded)
            raise
        previous_unload = await self.runtime.replace_loaded(model_name, loaded)
        self.active_state.set_active_version(model_name, version)
        await self.recorder.record(
            "MODEL_ACTIVATED",
            model=model_name,
            from_version=previous,
            to_version=version,
            healthcheck=healthcheck,
            previous_unload=previous_unload,
        )
        return {
            "model_name": model_name,
            "active": version,
            "previous": previous,
            "healthcheck": healthcheck,
            "previous_unload": previous_unload,
        }

    async def deactivate(self, model_name: str) -> dict[str, Any]:
        previous = self.active_state.deactivate(model_name)
        unload_info = await self.runtime.unload(model_name)
        await self.recorder.record(
            "MODEL_DEACTIVATED", model=model_name, previous=previous, unload=unload_info
        )
        return {
            "model_name": model_name,
            "active": None,
            "previous": previous,
            "unload": unload_info,
        }

    async def rollback(self, model_name: str) -> dict[str, Any]:
        current = self.active_state.get_active_version(model_name)
        previous = self.active_state.get_previous_version(model_name)
        if not previous:
            raise VersionNotFoundError(
                f"Model '{model_name}' has no previous version to rollback to"
            )
        previous_unload = await self._load(model_name, previous)
        self.active_state.swap_active_previous(model_name)
        await self.recorder.record(
            "MODEL_ROLLBACK",
            model=model_name,
            from_version=current,
            to_version=previous,
            previous_unload=previous_unload,
        )
        return {
            "model_name": model_name,
            "active": previous,
            "previous": current,
            "previous_unload": previous_unload,
        }

    async def _load(self, model_name: str, version: str) -> dict[str, Any]:
        metadata = self.registry.get_metadata(model_name, version)
        artifact_path = self.registry.get_artifact_path(model_name, version)
        return await self.runtime.load(model_name, version, artifact_path, metadata)

    async def _run_healthcheck(
        self, loaded: LoadedModel, metadata: ModelMetadata
    ) -> dict[str, Any]:
        inputs, payload = self._dummy_inputs(metadata)
        output_names = [spec.name for spec in metadata.outputs]
        if not output_names:
            raise ModelHealthCheckError(
                f"Model {loaded.name}:{loaded.version} declares no outputs"
            )
        try:
            output_headers, _output_payload, timings_ms = (
                await self.runtime.infer_loaded(loaded, inputs, payload, output_names)
            )
        except ServingError as exc:
            await self.recorder.record(
                "MODEL_HEALTHCHECK_FAILED",
                model=loaded.name,
                version=loaded.version,
                error_code=exc.code,
                error=exc.message,
            )
            raise ModelHealthCheckError(
                f"Healthcheck failed for {loaded.name}:{loaded.version}: {exc.message}"
            ) from exc

        returned_outputs = {item.get("name") for item in output_headers}
        missing_outputs = sorted(set(output_names) - returned_outputs)
        if missing_outputs:
            message = f"Healthcheck output is missing model outputs: {missing_outputs}"
            await self.recorder.record(
                "MODEL_HEALTHCHECK_FAILED",
                model=loaded.name,
                version=loaded.version,
                error=message,
            )
            raise ModelHealthCheckError(message)

        result = {
            "mode": "smoke",
            "status": "passed",
            "outputs": len(output_headers),
            "timings_ms": timings_ms,
        }
        await self.recorder.record(
            "MODEL_HEALTHCHECK_PASSED",
            model=loaded.name,
            version=loaded.version,
            **result,
        )
        return result

    def _dummy_inputs(
        self, metadata: ModelMetadata
    ) -> tuple[list[dict[str, Any]], bytes]:
        headers: list[dict[str, Any]] = []
        payload_parts: list[bytes] = []
        for spec in metadata.inputs:
            dtype = DATATYPE_TO_DTYPE.get(spec.datatype)
            if dtype is None:
                raise ModelHealthCheckError(
                    f"Unsupported healthcheck input datatype '{spec.datatype}'"
                )
            shape = [dim if dim > 0 else 1 for dim in spec.shape]
            array = np.zeros(shape, dtype=dtype)
            data = array.tobytes(order="C")
            payload_parts.append(data)
            headers.append(
                {
                    "name": spec.name,
                    "shape": shape,
                    "datatype": spec.datatype,
                    "parameters": {"binary_data_size": len(data)},
                }
            )
        if not headers:
            raise ModelHealthCheckError(
                f"Model {metadata.name}:{metadata.version} declares no inputs"
            )
        return headers, b"".join(payload_parts)
