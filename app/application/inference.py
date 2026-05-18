from __future__ import annotations

import cProfile
import os
import threading
from typing import Any

from app.common.exceptions import ModelNotReadyError, ServingError
from app.data_plane.tensor_inference.contract import (
    make_binary_response_header,
    validate_inputs,
    validate_output_names,
)
from app.data_plane.tensor_inference.observer import InferenceTelemetryObserver
from app.data_plane.worker_runtime.interfaces import RuntimeSupervisor


class InferenceService:
    def __init__(
        self,
        runtime: RuntimeSupervisor,
        observer: InferenceTelemetryObserver,
    ) -> None:
        self._runtime = runtime
        self._observer = observer
        self._profile_path = os.environ.get("SERVING_CPROFILE_PATH")
        self._profile = cProfile.Profile() if self._profile_path else None
        self._profile_lock = threading.Lock()

    async def infer(
        self, model_name: str, header: dict[str, Any], payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        if self._profile is not None:
            return await self._profiled_infer(model_name, header, payload)
        return await self._infer(model_name, header, payload)

    async def _profiled_infer(
        self, model_name: str, header: dict[str, Any], payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        assert self._profile is not None
        with self._profile_lock:
            self._profile.enable()
        try:
            return await self._infer(model_name, header, payload)
        finally:
            with self._profile_lock:
                self._profile.disable()
                if self._profile_path:
                    self._profile.dump_stats(self._profile_path)

    async def _infer(
        self, model_name: str, header: dict[str, Any], binary_payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        loaded = await self._runtime.get_loaded(model_name)
        if not loaded:
            raise ModelNotReadyError(f"Model '{model_name}' is not loaded")

        request_id = header.get("id")
        span = self._observer.start(model_name, loaded.version, request_id)
        try:
            inputs = validate_inputs(header, binary_payload, loaded.metadata)
            output_names = validate_output_names(header, loaded.metadata)
            output_headers, response_binary, timings_ms = (
                await self._runtime.infer_tensors(
                    model_name,
                    inputs,
                    binary_payload,
                    output_names,
                )
            )
            await self._observer.record_success(span, timings_ms)
            return (
                make_binary_response_header(
                    request_id, model_name, loaded.version, output_headers
                ),
                response_binary,
            )
        except ServingError as exc:
            await self._observer.record_error(span, exc.code)
            raise
