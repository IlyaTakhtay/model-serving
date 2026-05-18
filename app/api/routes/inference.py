from __future__ import annotations

from litestar import Request, post
from litestar.response import Response

from app.api.errors import raise_recorded_error
from app.api.protocols.binary_tensor import (
    make_binary_tensor_response,
    read_binary_tensor_request,
)
from app.common.exceptions import ServingError
from app.container import ApplicationContainer


@post("/v1/models/{model_name:str}/infer")
async def infer(
    model_name: str,
    request: Request,
    container: ApplicationContainer,
) -> Response[bytes]:
    try:
        tensor_request = await read_binary_tensor_request(request)
        response_header, response_payload = await container.inference.infer(
            model_name,
            tensor_request.header,
            tensor_request.payload,
        )
        return make_binary_tensor_response(response_header, response_payload)
    except ServingError as exc:
        await raise_recorded_error(container, exc, "REQUEST_FAILED", model=model_name)


route_handlers = [infer]
