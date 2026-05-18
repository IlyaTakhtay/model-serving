from __future__ import annotations

import os

import uvicorn
from litestar import Litestar

from .api.routes import dependencies, route_handlers
from .config.settings import Settings
from .container import ApplicationContainer

settings = Settings.from_env()


async def on_startup(app: Litestar) -> None:
    container = ApplicationContainer.build(settings)
    app.state.container = container
    await container.background.startup()


async def on_shutdown(app: Litestar) -> None:
    container = getattr(app.state, "container", None)
    if container is not None:
        await container.background.shutdown()


app = Litestar(
    route_handlers=route_handlers,
    dependencies=dependencies,
    on_startup=[on_startup],
    on_shutdown=[on_shutdown],
    request_max_body_size=settings.request_max_body_size,
)


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        access_log=_env_bool("SERVING_UVICORN_ACCESS_LOG", True),
        log_level=os.getenv("SERVING_UVICORN_LOG_LEVEL", "info"),
        loop=os.getenv("SERVING_UVICORN_LOOP", "auto"),
        http=os.getenv("SERVING_UVICORN_HTTP", "auto"),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
