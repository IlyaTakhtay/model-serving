from __future__ import annotations

from dataclasses import dataclass

from .api.protocols.binary_tensor import BinaryTensorProtocol
from .application.background import BackgroundApplication
from .application.inference_serving import TensorInferenceApplication
from .application.model_serving import ModelServingApplication
from .application.observability import ObservabilityApplication
from .config.settings import Settings
from .control_plane.model_management.active_state import ActiveModelStateStore
from .control_plane.model_management.artifact_inspector import OnnxArtifactInspector
from .control_plane.model_management.catalog import ModelCatalog
from .control_plane.model_management.lifecycle import ModelLifecycle
from .control_plane.model_management.registry import ModelRegistry
from .control_plane.model_management.uploader import ModelUploader
from .control_plane.rollback.auto_evaluator import AutoRollbackEvaluator
from .control_plane.rollback.monitor import AutoRollbackMonitor
from .control_plane.rollback.policy_manager import AutoRollbackPolicyManager
from .data_plane.tensor_inference.inferencer import TensorInferencer
from .data_plane.worker_runtime.runtime import WorkerRuntime
from .observability.exporters.prometheus import PrometheusMetricsExporter
from .observability.producers.inference import InferenceTelemetryRecorder
from .observability.producers.resources import ResourceSampler
from .observability.recorder import ObservabilityRecorder
from .observability.state.memory import InMemoryObservabilityState
from .observability.storage.ring import RingEventStorage
from .observability.writer import ObservabilityWriter


@dataclass
class ApplicationContainer:
    settings: Settings
    models: ModelServingApplication
    inference: TensorInferenceApplication
    observability: ObservabilityApplication
    background: BackgroundApplication

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
        observability_writer = ObservabilityWriter(
            event_storage,
            observability_state,
            queue_size=settings.observability.queue_size,
        )
        observability_recorder = ObservabilityRecorder(observability_writer)
        observability_recorder.replay_state(
            limit=settings.observability.replay_records
        )
        prometheus_exporter = PrometheusMetricsExporter()
        runtime = WorkerRuntime(cpu_budget=settings.cpu_budget)
        registry = ModelRegistry(settings.model_root)
        active_state = ActiveModelStateStore(settings.config_path)
        lifecycle = ModelLifecycle(
            registry,
            active_state,
            runtime,
            observability_recorder,
        )
        artifact_inspector = OnnxArtifactInspector()
        uploader = ModelUploader(
            registry,
            lifecycle,
            artifact_inspector,
            settings.upload_tmp_root,
            observability_recorder,
        )
        catalog = ModelCatalog(registry, active_state, runtime)
        rollback = AutoRollbackEvaluator(
            registry,
            active_state,
            runtime,
            observability_state,
            observability_recorder,
            lifecycle,
        )
        rollback_monitor = AutoRollbackMonitor(rollback, observability_recorder)
        inference_observer = InferenceTelemetryRecorder(
            runtime,
            observability_recorder,
            on_request_recorded=rollback_monitor.on_request_recorded,
        )
        inferencer = TensorInferencer(runtime, inference_observer)
        inference = TensorInferenceApplication(BinaryTensorProtocol(), inferencer)
        rollback_policies = AutoRollbackPolicyManager(
            registry, active_state, observability_recorder
        )
        resource_sampler = ResourceSampler(
            runtime,
            observability_recorder,
            settings.resource_sampling_interval_sec,
            on_snapshot_recorded=rollback_monitor.on_resource_snapshot,
        )
        return cls(
            settings=settings,
            models=ModelServingApplication(
                catalog, lifecycle, uploader, rollback_policies
            ),
            inference=inference,
            observability=ObservabilityApplication(
                observability_recorder,
                event_storage,
                observability_state,
                prometheus_exporter,
                runtime,
            ),
            background=BackgroundApplication(
                lifecycle, runtime, resource_sampler, observability_recorder
            ),
        )
