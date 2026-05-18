from __future__ import annotations

from typing import Any

from litestar import get
from litestar.response import Response

from app.container import ApplicationContainer


@get("/v1/models/{model_name:str}/timings")
async def model_timings(
    model_name: str, container: ApplicationContainer, limit: int = 100
) -> dict[str, Any]:
    return container.observability.model_timings(model_name, limit)


@get("/v1/runtime/resources")
async def runtime_resources(container: ApplicationContainer) -> dict[str, Any]:
    return container.observability.runtime_resources()


@get("/metrics")
async def metrics(container: ApplicationContainer) -> Response[str]:
    return Response(
        content=container.observability.prometheus_metrics(),
        media_type="text/plain; version=0.0.4",
    )


@get("/events")
async def events(container: ApplicationContainer, limit: int = 100) -> dict[str, Any]:
    return container.observability.recent_events(limit)


route_handlers = [model_timings, runtime_resources, metrics, events]
