from __future__ import annotations

from typing import Any

from app.common.exceptions import ModelNotReadyError, ServingError
from app.data_plane.tensor_inference.contract import TensorContractValidator
from app.data_plane.tensor_inference.observer import InferenceTelemetryObserver
from app.data_plane.worker_runtime.runtime import WorkerRuntime


class TensorInferencer:
    def __init__(
        self,
        runtime: WorkerRuntime,
        observer: InferenceTelemetryObserver,
        tensor_contract: TensorContractValidator | None = None,
    ) -> None:
        self.runtime = runtime
        self.observer = observer
        self.tensor_contract = tensor_contract or TensorContractValidator()

    async def infer(
        self, model_name: str, header: dict[str, Any], binary_payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        loaded = await self.runtime.get_loaded(model_name)
        if not loaded:
            raise ModelNotReadyError(f"Model '{model_name}' is not loaded")

        request_id = header.get("id")
        span = self.observer.start(model_name, loaded.version, request_id)
        try:
            inputs = self.tensor_contract.validate_inputs(
                header, binary_payload, loaded.metadata
            )
            output_names = self.tensor_contract.validate_output_names(
                header, loaded.metadata
            )
            output_headers, response_binary, timings_ms = await self.runtime.infer_tensors(
                model_name,
                inputs,
                binary_payload,
                output_names,
            )
            await self.observer.record_success(span, timings_ms)
            return (
                self._make_binary_response_header(
                    header.get("id"), model_name, loaded.version, output_headers
                ),
                response_binary,
            )
        except ServingError as exc:
            await self.observer.record_error(span, exc.code)
            raise

    def _make_binary_response_header(
        self,
        request_id: str | None,
        model_name: str,
        model_version: str,
        outputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "model_name": model_name,
            "model_version": model_version,
            "outputs": outputs,
        }
