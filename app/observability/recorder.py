from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.observability.events import ObservabilityEvent, event_to_dict, make_event
from app.observability.state.memory import InMemoryObservabilityState
from app.observability.storage.ring import RingEventStorage
from app.observability.writer import ObservabilityWriter

EventListener = Callable[[ObservabilityEvent], Awaitable[None]]


class ObservabilityRecorder:
    def __init__(
        self,
        storage: RingEventStorage,
        state: InMemoryObservabilityState,
        *,
        queue_size: int = 10000,
    ) -> None:
        self._storage = storage
        self._state = state
        self._writer = ObservabilityWriter(storage, state, queue_size=queue_size)
        self._listeners: list[EventListener] = []

    def start(self) -> None:
        self._writer.start()

    async def stop(self) -> None:
        await self._writer.stop()

    def subscribe(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    async def record(self, event: str, **fields: Any) -> ObservabilityEvent:
        observation = make_event(event, **fields)
        written = await self._writer.write(observation)
        for listener in tuple(self._listeners):
            await listener(written)
        return written

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

    async def record_runtime_sample(
        self,
        *,
        cpu_budget_threads: int,
        cpu_requested_threads: int,
        loaded_models: dict[str, str],
    ) -> ObservabilityEvent:
        return await self.record(
            "RUNTIME_SAMPLED",
            cpu_budget_threads=cpu_budget_threads,
            cpu_requested_threads=cpu_requested_threads,
            cpu_available_threads=cpu_budget_threads - cpu_requested_threads,
            loaded_models=dict(loaded_models),
        )

    def replay_state(self, limit: int | None = None) -> None:
        self._writer.replay_state(limit=limit)

    def recent_timings(
        self, model_name: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self._state.recent_timings(model_name, limit)

    def request_history(
        self, model_name: str, version: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return self._state.request_history(model_name, version, limit)

    def clear_request_history(
        self, model_name: str, version: str | None = None
    ) -> None:
        self._state.clear_request_history(model_name, version)

    def latest_resources(self) -> dict[str, dict[str, Any]]:
        return self._state.latest_resources()

    def latest_runtime(self) -> dict[str, Any]:
        return self._state.latest_runtime()

    def resource_history(
        self,
        model_name: str,
        version: str | None = None,
        *,
        window_sec: float | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        return self._state.resource_history(
            model_name, version, window_sec=window_sec, limit=limit
        )

    def metrics_snapshot(self) -> dict[str, dict]:
        return self._state.metrics_snapshot()

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(0, min(limit, 1000))
        return [event_to_dict(event) for event in self._storage.read_latest(limit)]
