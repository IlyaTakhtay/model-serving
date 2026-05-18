from __future__ import annotations

from typing import Any

from litestar import get

from app.container import ApplicationContainer


@get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@get("/ready")
async def ready(container: ApplicationContainer) -> dict[str, Any]:
    loaded = container.model_control.loaded_models()
    return {"status": "ok" if loaded else "not_ready", "loaded_models": loaded}


@get("/v1")
async def v1() -> dict[str, list[str]]:
    return {
        "extensions": [
            "local-model-registry",
            "model-upload",
            "model-upload-chunk",
            "tensor-inference",
            "model-activate",
            "model-deactivate",
            "model-rollback",
            "auto-rollback-policy",
        ]
    }


route_handlers = [health, ready, v1]
