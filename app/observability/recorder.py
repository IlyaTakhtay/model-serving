from __future__ import annotations

from typing import Any

from app.observability.events import ObservabilityEvent, make_event
from app.observability.writer import ObservabilityWriter


class ObservabilityRecorder:
    def __init__(self, writer: ObservabilityWriter) -> None:
        self._writer = writer

    def start(self) -> None:
        self._writer.start()

    async def stop(self) -> None:
        await self._writer.stop()

    async def record(self, event: str, **fields: Any) -> ObservabilityEvent:
        observation = make_event(event, **fields)
        return await self._writer.write(observation)

    async def record_inference(
        self,
        *,
        model: str,
        version: str,
        status: str,
        latency_ms: float,
        request_id: str | None = None,
        timings_ms: dict[str, float] | None = None,
        resources: dict[str, float] | None = None,
        error_code: str | None = None,
    ) -> ObservabilityEvent:
        return await self.record(
            "INFERENCE" if status == "ok" else "INFERENCE_FAILED",
            model=model,
            version=version,
            request_id=request_id,
            status=status,
            latency_ms=latency_ms,
            timings_ms=timings_ms,
            resources=resources,
            error_code=error_code,
            protocol="binary",
        )

    async def record_resource_sample(
        self, model: str, snapshot: dict[str, Any]
    ) -> ObservabilityEvent:
        return await self.record(
            "RESOURCE_SAMPLED",
            model=model,
            version=snapshot.get("version"),
            pid=snapshot.get("pid"),
            status=snapshot.get("status"),
            memory_mb=snapshot.get("memory_mb"),
            cpu_seconds_total=snapshot.get("cpu_seconds_total"),
            cpu_usage_percent=snapshot.get("cpu_usage_percent"),
            threads=snapshot.get("threads"),
            requested_threads=snapshot.get("requested_threads"),
            worker_crashes=snapshot.get("worker_crashes"),
            worker_restart_failures=snapshot.get("worker_restart_failures"),
        )

    def replay_state(self, limit: int | None = None) -> None:
        self._writer.replay_state(limit=limit)
