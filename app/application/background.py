from __future__ import annotations

from app.control_plane.model_management.lifecycle import ModelLifecycle
from app.data_plane.worker_runtime.runtime import WorkerRuntime
from app.observability.producers.resources import ResourceSampler
from app.observability.recorder import ObservabilityRecorder


class BackgroundApplication:
    def __init__(
        self,
        lifecycle: ModelLifecycle,
        runtime: WorkerRuntime,
        resource_sampler: ResourceSampler,
        recorder: ObservabilityRecorder,
    ) -> None:
        self._lifecycle = lifecycle
        self._runtime = runtime
        self._resource_sampler = resource_sampler
        self._recorder = recorder

    async def startup(self) -> None:
        self._recorder.start()
        await self._lifecycle.load_active_models()
        self._resource_sampler.start()
        await self._recorder.record("SERVICE_STARTED")

    async def shutdown(self) -> None:
        await self._resource_sampler.stop()
        for model_name in list(self._runtime.loaded_models()):
            await self._runtime.unload(model_name)
        await self._recorder.record("SERVICE_STOPPED")
        await self._recorder.stop()
