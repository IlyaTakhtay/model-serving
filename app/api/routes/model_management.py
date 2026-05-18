from __future__ import annotations

from typing import Any

from litestar import delete, get, post, put

from app.api.errors import raise_recorded_error
from app.common.exceptions import ServingError
from app.container import ApplicationContainer
from app.schemas.upload import ModelUploadChunkRequest, ModelUploadRequest


@get("/v1/models")
async def list_models(container: ApplicationContainer) -> dict[str, Any]:
    try:
        return container.model_control.list_models()
    except ServingError as exc:
        await raise_recorded_error(container, exc)


@get("/v1/models/{model_name:str}")
async def describe_model(
    model_name: str, container: ApplicationContainer
) -> dict[str, Any]:
    try:
        return container.model_control.describe_model(model_name)
    except ServingError as exc:
        await raise_recorded_error(container, exc)


@post("/v1/models/{model_name:str}/versions/{version:str}/upload")
async def upload_model(
    model_name: str,
    version: str,
    data: ModelUploadRequest,
    container: ApplicationContainer,
) -> dict[str, Any]:
    try:
        return await container.model_control.upload(model_name, version, data)
    except ServingError as exc:
        await raise_recorded_error(
            container, exc, "MODEL_UPLOAD_FAILED", model=model_name, version=version
        )


@post("/v1/models/{model_name:str}/versions/{version:str}/upload-chunk")
async def upload_model_chunk(
    model_name: str,
    version: str,
    data: ModelUploadChunkRequest,
    container: ApplicationContainer,
) -> dict[str, Any]:
    try:
        return await container.model_control.upload_chunk(model_name, version, data)
    except ServingError as exc:
        await raise_recorded_error(
            container,
            exc,
            "MODEL_UPLOAD_CHUNK_FAILED",
            model=model_name,
            version=version,
        )


@post("/v1/models/{model_name:str}/versions/{version:str}/activate")
async def activate_model(
    model_name: str, version: str, container: ApplicationContainer
) -> dict[str, Any]:
    try:
        return await container.model_control.activate(model_name, version)
    except ServingError as exc:
        await raise_recorded_error(
            container, exc, "MODEL_ACTIVATE_FAILED", model=model_name, version=version
        )


@post("/v1/models/{model_name:str}/deactivate")
async def deactivate_model(
    model_name: str, container: ApplicationContainer
) -> dict[str, Any]:
    try:
        return await container.model_control.deactivate(model_name)
    except ServingError as exc:
        await raise_recorded_error(
            container, exc, "MODEL_DEACTIVATE_FAILED", model=model_name
        )


@post("/v1/models/{model_name:str}/rollback")
async def rollback_model(
    model_name: str, container: ApplicationContainer
) -> dict[str, Any]:
    try:
        return await container.model_control.rollback(model_name)
    except ServingError as exc:
        await raise_recorded_error(
            container, exc, "MODEL_ROLLBACK_FAILED", model=model_name
        )


@get("/v1/models/{model_name:str}/rollback-policy")
async def get_active_rollback_policy(
    model_name: str, container: ApplicationContainer
) -> dict[str, Any]:
    try:
        return container.model_control.get_active_rollback_policy(model_name)
    except ServingError as exc:
        await raise_recorded_error(
            container, exc, "ROLLBACK_POLICY_READ_FAILED", model=model_name
        )


@put("/v1/models/{model_name:str}/rollback-policy")
async def set_active_rollback_policy(
    model_name: str,
    data: dict[str, Any],
    container: ApplicationContainer,
) -> dict[str, Any]:
    try:
        return await container.model_control.set_active_rollback_policy(model_name, data)
    except ServingError as exc:
        await raise_recorded_error(
            container, exc, "ROLLBACK_POLICY_UPDATE_FAILED", model=model_name
        )


@delete("/v1/models/{model_name:str}/rollback-policy", status_code=200)
async def disable_active_rollback_policy(
    model_name: str, container: ApplicationContainer
) -> dict[str, Any]:
    try:
        return await container.model_control.disable_active_rollback_policy(model_name)
    except ServingError as exc:
        await raise_recorded_error(
            container, exc, "ROLLBACK_POLICY_DISABLE_FAILED", model=model_name
        )


@get("/v1/models/{model_name:str}/versions/{version:str}/rollback-policy")
async def get_version_rollback_policy(
    model_name: str,
    version: str,
    container: ApplicationContainer,
) -> dict[str, Any]:
    try:
        return container.model_control.get_version_rollback_policy(model_name, version)
    except ServingError as exc:
        await raise_recorded_error(
            container,
            exc,
            "ROLLBACK_POLICY_READ_FAILED",
            model=model_name,
            version=version,
        )


@put("/v1/models/{model_name:str}/versions/{version:str}/rollback-policy")
async def set_version_rollback_policy(
    model_name: str,
    version: str,
    data: dict[str, Any],
    container: ApplicationContainer,
) -> dict[str, Any]:
    try:
        return await container.model_control.set_version_rollback_policy(
            model_name, version, data
        )
    except ServingError as exc:
        await raise_recorded_error(
            container,
            exc,
            "ROLLBACK_POLICY_UPDATE_FAILED",
            model=model_name,
            version=version,
        )


@delete(
    "/v1/models/{model_name:str}/versions/{version:str}/rollback-policy",
    status_code=200,
)
async def disable_version_rollback_policy(
    model_name: str,
    version: str,
    container: ApplicationContainer,
) -> dict[str, Any]:
    try:
        return await container.model_control.disable_version_rollback_policy(
            model_name, version
        )
    except ServingError as exc:
        await raise_recorded_error(
            container,
            exc,
            "ROLLBACK_POLICY_DISABLE_FAILED",
            model=model_name,
            version=version,
        )


route_handlers = [
    list_models,
    describe_model,
    upload_model,
    upload_model_chunk,
    activate_model,
    deactivate_model,
    rollback_model,
    get_active_rollback_policy,
    set_active_rollback_policy,
    disable_active_rollback_policy,
    get_version_rollback_policy,
    set_version_rollback_policy,
    disable_version_rollback_policy,
]
