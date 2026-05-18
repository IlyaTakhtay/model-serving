from __future__ import annotations

import os

import uvicorn

from .api.rest import LitestarRestTransport
from .config.settings import Settings

settings = Settings.from_env()
app = LitestarRestTransport(settings).build_app()


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
