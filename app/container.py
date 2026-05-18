from __future__ import annotations

from dataclasses import dataclass

from .application.inference import InferenceService
from .application.model_control import ModelControlService
from .application.observability import ObservabilityService
from .application.service_lifecycle import ServiceLifecycle
from .config.settings import Settings
from .control_plane.model_management.active_state import ActiveModelStateStore
from .control_plane.model_management.artifact_inspector import (
    ArtifactInspectorRegistry,
    OnnxArtifactInspector,
)
from .control_plane.model_management.catalog import ModelCatalog
from .control_plane.model_management.lifecycle import ModelLifecycle
from .control_plane.model_management.registry import ModelRegistry
from .control_plane.model_management.uploader import ModelUploader
from .control_plane.rollback.auto_evaluator import AutoRollbackEvaluator
from .control_plane.rollback.monitor import AutoRollbackMonitor
from .control_plane.rollback.policy_manager import AutoRollbackPolicyManager
from .data_plane.worker_runtime.runtime import WorkerRuntime
from .observability.producers.inference import InferenceTelemetryRecorder
from .observability.producers.resources import ResourceSampler
from .observability.recorder import ObservabilityRecorder
from .observability.state.memory import InMemoryObservabilityState
from .observability.storage.ring import RingEventStorage


@dataclass
class ApplicationContainer:
    settings: Settings
    model_control: ModelControlService
    inference: InferenceService
    observability: ObservabilityService
    lifecycle: ServiceLifecycle

    @classmethod
    def build(cls, settings: Settings) -> ApplicationContainer:
        observability_state = InMemoryObservabilityState(
            recent_timings_maxlen=settings.observability.recent_timings_maxlen,
            request_window_maxlen=settings.observability.request_window_maxlen,
            resource_window_maxlen=settings.observability.resource_window_maxlen,
        )
        event_storage = RingEventStorage(
            settings.observability.ring_path,
            settings.observability.ring_size_bytes,
        )
        observability_recorder = ObservabilityRecorder(
            event_storage,
            observability_state,
            queue_size=settings.observability.queue_size,
        )
        observability_recorder.replay_state(
            limit=settings.observability.replay_records
        )
        runtime = WorkerRuntime(cpu_budget=settings.cpu_budget)
        registry = ModelRegistry(settings.model_root)
        active_state = ActiveModelStateStore(settings.config_path)
        lifecycle = ModelLifecycle(
            registry,
            active_state,
            runtime,
            observability_recorder,
        )
        artifact_inspectors = ArtifactInspectorRegistry([OnnxArtifactInspector()])
        uploader = ModelUploader(
            registry,
            lifecycle,
            artifact_inspectors,
            settings.upload_tmp_root,
            observability_recorder,
        )
        catalog = ModelCatalog(registry, active_state, runtime)
        rollback = AutoRollbackEvaluator(
            registry,
            active_state,
            runtime,
            observability_recorder,
            lifecycle,
        )
        rollback_monitor = AutoRollbackMonitor(rollback, observability_recorder)
        observability_recorder.subscribe(rollback_monitor.on_observability_event)
        inference_observer = InferenceTelemetryRecorder(
            runtime,
            observability_recorder,
        )
        rollback_policies = AutoRollbackPolicyManager(
            registry, active_state, observability_recorder
        )
        resource_sampler = ResourceSampler(
            runtime,
            observability_recorder,
            settings.resource_sampling_interval_sec,
        )
        return cls(
            settings=settings,
            model_control=ModelControlService(
                catalog, lifecycle, uploader, rollback_policies
            ),
            inference=InferenceService(runtime, inference_observer),
            observability=ObservabilityService(
                observability_recorder,
            ),
            lifecycle=ServiceLifecycle(
                lifecycle, runtime, resource_sampler, observability_recorder
            ),
        )
