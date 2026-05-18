from __future__ import annotations

from litestar import Request, post
from litestar.response import Response

from app.api.errors import raise_recorded_error
from app.common.exceptions import ServingError
from app.container import ApplicationContainer


@post("/v1/models/{model_name:str}/infer")
async def infer(
    model_name: str, request: Request, container: ApplicationContainer
) -> Response[bytes]:
    try:
        return await container.inference.infer_binary(model_name, request)
    except ServingError as exc:
        raise_recorded_error(container, exc, "REQUEST_FAILED", model=model_name)


route_handlers = [infer]
