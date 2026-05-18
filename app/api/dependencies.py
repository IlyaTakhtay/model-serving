from __future__ import annotations

from litestar.datastructures import State
from litestar.di import Provide

from app.container import ApplicationContainer


def provide_container(state: State) -> ApplicationContainer:
    container = getattr(state, "container", None)
    if not isinstance(container, ApplicationContainer):
        raise RuntimeError("Application container is not initialized")
    return container


dependencies = {"container": Provide(provide_container, sync_to_thread=False)}
