from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InferenceTelemetrySpan:
    model_name: str
    version: str
    request_id: str | None
    started_at: float
    resources_before: dict[str, float]


class InferenceTelemetryObserver(Protocol):
    def start(
        self, model_name: str, version: str, request_id: str | None
    ) -> InferenceTelemetrySpan: ...

    async def record_success(
        self, span: InferenceTelemetrySpan, timings_ms: dict[str, float]
    ) -> None: ...

    async def record_error(
        self, span: InferenceTelemetrySpan, error_code: str
    ) -> None: ...
