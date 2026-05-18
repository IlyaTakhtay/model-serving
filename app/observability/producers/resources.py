from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from app.data_plane.worker_runtime.interfaces import RuntimeSupervisor
from app.observability.recorder import ObservabilityRecorder


class ResourceSampler:
    def __init__(
        self,
        runtime: RuntimeSupervisor,
        recorder: ObservabilityRecorder,
        interval_sec: float,
    ) -> None:
        self.runtime = runtime
        self.recorder = recorder
        self.interval_sec = max(1.0, interval_sec)
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._previous_cpu: dict[tuple[str, int], tuple[float, float]] = {}

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="resource-sampler")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stopping:
            await self.sample_once()
            await asyncio.sleep(self.interval_sec)

    async def sample_once(self) -> None:
        try:
            now = time.time()
            await self.recorder.record_runtime_sample(
                cpu_budget_threads=self.runtime.cpu_budget,
                cpu_requested_threads=self.runtime.cpu_requested_threads(),
                loaded_models=self.runtime.loaded_models(),
            )
            for model_name, data in self.runtime.process_metrics().items():
                snapshot = dict(data)
                snapshot["cpu_usage_percent"] = self._cpu_usage_percent(
                    model_name, snapshot, now
                )
                await self.recorder.record_resource_sample(model_name, snapshot)
        except Exception as exc:
            await self.recorder.record("RESOURCE_SAMPLER_FAILED", error=str(exc))

    def _cpu_usage_percent(
        self, model_name: str, data: dict, now: float
    ) -> float | None:
        pid = data.get("pid")
        cpu_seconds = data.get("cpu_seconds_total")
        if pid is None or cpu_seconds is None:
            return None
        key = (model_name, int(pid))
        current = float(cpu_seconds)
        previous = self._previous_cpu.get(key)
        self._previous_cpu[key] = (now, current)
        if previous is None:
            return None
        previous_ts, previous_cpu = previous
        elapsed = now - previous_ts
        if elapsed <= 0:
            return None
        return max(0.0, current - previous_cpu) / elapsed * 100.0
