from __future__ import annotations

from app.control_plane.model_management.lifecycle import ModelLifecycle
from app.data_plane.worker_runtime.interfaces import RuntimeSupervisor
from app.observability.producers.resources import ResourceSampler
from app.observability.recorder import ObservabilityRecorder


class ServiceLifecycle:
    def __init__(
        self,
        model_lifecycle: ModelLifecycle,
        runtime: RuntimeSupervisor,
        resource_sampler: ResourceSampler,
        recorder: ObservabilityRecorder,
    ) -> None:
        self._model_lifecycle = model_lifecycle
        self._runtime = runtime
        self._resource_sampler = resource_sampler
        self._recorder = recorder

    async def startup(self) -> None:
        self._recorder.start()
        await self._model_lifecycle.load_active_models()
        await self._resource_sampler.sample_once()
        self._resource_sampler.start()
        await self._recorder.record("SERVICE_STARTED")

    async def shutdown(self) -> None:
        await self._resource_sampler.stop()
        await self._runtime.shutdown_all()
        await self._recorder.record("SERVICE_STOPPED")
        await self._recorder.stop()
