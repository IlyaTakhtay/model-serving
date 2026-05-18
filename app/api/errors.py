from __future__ import annotations

from typing import Any, NoReturn

from litestar.exceptions import HTTPException

from app.common.exceptions import ServingError
from app.container import ApplicationContainer


def to_http_error(exc: ServingError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "error": {"code": exc.code, "message": exc.message, "layer": exc.layer}
        },
    )


def raise_recorded_error(
    container: ApplicationContainer,
    exc: ServingError,
    event: str | None = None,
    **event_fields: Any,
) -> NoReturn:
    container.observability.record_error(
        exc.code, event, **event_fields, error_code=exc.code, error=exc.message
    )
    raise to_http_error(exc)
