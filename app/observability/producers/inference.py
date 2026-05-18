from __future__ import annotations

import time

from app.data_plane.tensor_inference.observer import InferenceTelemetrySpan
from app.data_plane.worker_runtime.interfaces import RuntimeSupervisor
from app.observability.recorder import ObservabilityRecorder


class InferenceTelemetryRecorder:
    def __init__(
        self,
        runtime: RuntimeSupervisor,
        recorder: ObservabilityRecorder,
    ) -> None:
        self.runtime = runtime
        self.recorder = recorder

    def start(
        self, model_name: str, version: str, request_id: str | None
    ) -> InferenceTelemetrySpan:
        return InferenceTelemetrySpan(
            model_name=model_name,
            version=version,
            request_id=request_id,
            started_at=time.perf_counter(),
            resources_before=self._sample_model_resources(model_name),
        )

    async def record_success(
        self, span: InferenceTelemetrySpan, timings_ms: dict[str, float]
    ) -> None:
        total_ms = self._elapsed_ms(span)
        recorded_timings = dict(timings_ms)
        recorded_timings["request"] = total_ms
        resources = self._resource_delta(span, total_ms)
        await self.recorder.record_inference(
            model=span.model_name,
            version=span.version,
            status="ok",
            latency_ms=total_ms,
            request_id=span.request_id,
            timings_ms=recorded_timings,
            resources=resources,
        )

    async def record_error(self, span: InferenceTelemetrySpan, error_code: str) -> None:
        total_ms = self._elapsed_ms(span)
        resources = self._resource_delta(span, total_ms)
        await self.recorder.record_inference(
            model=span.model_name,
            version=span.version,
            status="error",
            latency_ms=total_ms,
            request_id=span.request_id,
            resources=resources,
            error_code=error_code,
        )

    def _elapsed_ms(self, span: InferenceTelemetrySpan) -> float:
        return (time.perf_counter() - span.started_at) * 1000.0

    def _sample_model_resources(self, model_name: str) -> dict[str, float]:
        data = self.runtime.process_metrics().get(model_name, {})
        if data.get("status") != "running":
            return {}
        return {
            key: float(data[key])
            for key in ("memory_mb", "cpu_seconds_total")
            if key in data
        }

    def _resource_delta(
        self, span: InferenceTelemetrySpan, elapsed_ms: float
    ) -> dict[str, float]:
        after = self._sample_model_resources(span.model_name)
        resources = {"memory_mb": after["memory_mb"]} if "memory_mb" in after else {}
        if (
            "cpu_seconds_total" in span.resources_before
            and "cpu_seconds_total" in after
            and elapsed_ms > 0
        ):
            cpu_seconds = max(
                0.0,
                after["cpu_seconds_total"] - span.resources_before["cpu_seconds_total"],
            )
            resources["cpu_usage_percent"] = cpu_seconds / (elapsed_ms / 1000.0) * 100.0
        return resources
