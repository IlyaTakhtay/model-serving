from __future__ import annotations

from litestar import Litestar

from app.api.routes import dependencies, route_handlers
from app.config.settings import Settings
from app.container import ApplicationContainer


class LitestarRestTransport:
    """REST transport adapter for the serving application."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_app(self) -> Litestar:
        return Litestar(
            route_handlers=route_handlers,
            dependencies=dependencies,
            on_startup=[self._on_startup],
            on_shutdown=[self._on_shutdown],
            request_max_body_size=self._settings.request_max_body_size,
        )

    async def _on_startup(self, app: Litestar) -> None:
        container = ApplicationContainer.build(self._settings)
        app.state.container = container
        await container.lifecycle.startup()

    async def _on_shutdown(self, app: Litestar) -> None:
        container = getattr(app.state, "container", None)
        if container is not None:
            await container.lifecycle.shutdown()
