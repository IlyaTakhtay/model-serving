from __future__ import annotations

import cProfile
import os
import threading

from litestar import Request
from litestar.response import Response

from app.api.protocols.binary_tensor import BinaryTensorProtocol
from app.data_plane.tensor_inference.inferencer import TensorInferencer


class TensorInferenceApplication:
    def __init__(
        self, protocol: BinaryTensorProtocol, inferencer: TensorInferencer
    ) -> None:
        self._protocol = protocol
        self._inferencer = inferencer
        self._profile_path = os.environ.get("SERVING_CPROFILE_PATH")
        self._profile = cProfile.Profile() if self._profile_path else None
        self._profile_lock = threading.Lock()

    async def infer_binary(self, model_name: str, request: Request) -> Response[bytes]:
        if self._profile is not None:
            return await self._profiled_infer_binary(model_name, request)
        return await self._infer_binary(model_name, request)

    async def _infer_binary(self, model_name: str, request: Request) -> Response[bytes]:
        tensor_request = await self._protocol.read_request(request)
        response_header, response_binary = await self._inferencer.infer(
            model_name,
            tensor_request.header,
            tensor_request.payload,
        )
        return self._protocol.make_response(response_header, response_binary)

    async def _profiled_infer_binary(
        self, model_name: str, request: Request
    ) -> Response[bytes]:
        assert self._profile is not None
        with self._profile_lock:
            self._profile.enable()
        try:
            return await self._infer_binary(model_name, request)
        finally:
            with self._profile_lock:
                self._profile.disable()
                if self._profile_path:
                    self._profile.dump_stats(self._profile_path)
