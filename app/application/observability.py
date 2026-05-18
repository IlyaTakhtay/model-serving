from __future__ import annotations

from typing import Any

from app.data_plane.worker_runtime.runtime import WorkerRuntime
from app.observability.events import event_to_dict
from app.observability.exporters.prometheus import PrometheusMetricsExporter
from app.observability.recorder import ObservabilityRecorder
from app.observability.state.memory import InMemoryObservabilityState
from app.observability.storage.ring import RingEventStorage


class ObservabilityApplication:
    def __init__(
        self,
        recorder: ObservabilityRecorder,
        storage: RingEventStorage,
        state: InMemoryObservabilityState,
        prometheus_exporter: PrometheusMetricsExporter,
        runtime: WorkerRuntime,
    ) -> None:
        self._recorder = recorder
        self._storage = storage
        self._state = state
        self._prometheus_exporter = prometheus_exporter
        self._runtime = runtime

    async def record_error(
        self, code: str, event: str | None = None, **event_fields: Any
    ) -> None:
        await self._recorder.record(
            event or "REQUEST_ERROR",
            error_code=code,
            **event_fields,
        )

    def model_timings(self, model_name: str, limit: int = 100) -> dict[str, Any]:
        return {"timings": self._state.recent_timings(model_name, limit)}

    def runtime_resources(self) -> dict[str, Any]:
        processes = self._state.latest_resources() or self._runtime.process_metrics()
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
        cpu_requested = self._runtime.cpu_requested_threads()
        return {
            "cpu_budget_threads": self._runtime.cpu_budget,
            "cpu_requested_threads": cpu_requested,
            "cpu_available_threads": self._runtime.cpu_budget - cpu_requested,
            "models": models,
        }

    def prometheus_metrics(self) -> str:
        return self._prometheus_exporter.render(
            self._state.metrics_snapshot(),
            self._runtime.loaded_models(),
            self._runtime.process_metrics(),
            cpu_budget=self._runtime.cpu_budget,
            cpu_requested=self._runtime.cpu_requested_threads(),
        )

    def recent_events(self, limit: int = 100) -> dict[str, Any]:
        limit = max(0, min(limit, 1000))
        return {
            "events": [
                event_to_dict(event) for event in self._storage.read_latest(limit)
            ]
        }

    async def write_event(self, event: str, **fields: Any) -> None:
        await self._recorder.record(event, **fields)
