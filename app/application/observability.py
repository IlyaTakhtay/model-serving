from __future__ import annotations

from typing import Any

from app.observability.exporters.prometheus import render_prometheus_metrics
from app.observability.recorder import ObservabilityRecorder


class ObservabilityService:
    def __init__(
        self,
        recorder: ObservabilityRecorder,
    ) -> None:
        self._recorder = recorder

    async def record_error(
        self, code: str, event: str | None = None, **event_fields: Any
    ) -> None:
        await self._recorder.record(
            event or "REQUEST_ERROR",
            error_code=code,
            **event_fields,
        )

    def model_timings(self, model_name: str, limit: int = 100) -> dict[str, Any]:
        return {"timings": self._recorder.recent_timings(model_name, limit)}

    def runtime_resources(self) -> dict[str, Any]:
        runtime = self._recorder.latest_runtime()
        processes = self._recorder.latest_resources()
        models = [
            {
                "model": model,
                "version": data.get("version"),
                "pid": data.get("pid"),
                "status": data.get("status"),
                "requested_threads": data.get("requested_threads", 0),
                "memory_mb": data.get("memory_mb"),
                "cpu_seconds_total": data.get("cpu_seconds_total"),
                "cpu_usage_percent": data.get("cpu_usage_percent"),
                "worker_crashes": data.get("worker_crashes", 0),
                "worker_restart_failures": data.get("worker_restart_failures", 0),
                "sampled_at": data.get("ts"),
            }
            for model, data in sorted(processes.items())
        ]
        return {
            "cpu_budget_threads": int(runtime.get("cpu_budget_threads", 0)),
            "cpu_requested_threads": int(runtime.get("cpu_requested_threads", 0)),
            "cpu_available_threads": int(runtime.get("cpu_available_threads", 0)),
            "models": models,
        }

    def prometheus_metrics(self) -> str:
        runtime = self._recorder.latest_runtime()
        processes = self._recorder.latest_resources()
        loaded_models = runtime.get("loaded_models") or {
            model: str(data["version"])
            for model, data in processes.items()
            if data.get("status") == "running" and data.get("version") is not None
        }
        return render_prometheus_metrics(
            self._recorder.metrics_snapshot(),
            dict(loaded_models),
            processes,
            cpu_budget=runtime.get("cpu_budget_threads"),
            cpu_requested=runtime.get("cpu_requested_threads"),
        )

    def recent_events(self, limit: int = 100) -> dict[str, Any]:
        return {"events": self._recorder.recent_events(limit)}
