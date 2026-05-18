from __future__ import annotations

from typing import Any, Protocol


class ApiTransport(Protocol):
    """Builds an executable transport adapter for the application."""

    def build_app(self) -> Any:
        """Return a framework-specific application object."""
