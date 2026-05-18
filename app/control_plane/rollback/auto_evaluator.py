from __future__ import annotations

from typing import Any

from app.common.exceptions import ModelNotReadyError
from app.control_plane.model_management.active_state import ActiveModelStateStore
from app.control_plane.model_management.lifecycle import ModelLifecycle
from app.control_plane.model_management.registry import ModelRegistry
from app.data_plane.worker_runtime.runtime import WorkerRuntime
from app.domain.rollback import RollbackPolicy, RollbackStats, RollbackViolation
from app.observability.recorder import ObservabilityRecorder
from app.observability.state.memory import InMemoryObservabilityState


class AutoRollbackEvaluator:
    def __init__(
        self,
        registry: ModelRegistry,
        active_state: ActiveModelStateStore,
        runtime: WorkerRuntime,
        observability_state: InMemoryObservabilityState,
        observability_recorder: ObservabilityRecorder,
        lifecycle: ModelLifecycle,
    ) -> None:
        self.registry = registry
        self.active_state = active_state
        self.runtime = runtime
        self.observability_state = observability_state
        self.observability_recorder = observability_recorder
        self.lifecycle = lifecycle

    async def evaluate(self, model_name: str, trigger: str) -> dict[str, Any]:
        active = self.active_state.get_active_version(model_name)
        if not active:
            raise ModelNotReadyError(f"Model '{model_name}' has no active version")

        metadata = self.registry.get_metadata(model_name, active)
        policy = RollbackPolicy.from_dict(metadata.rollback_policy)
        if not policy.enabled:
            return self._result(
                model_name, active, policy, "disabled", [], {"requests": 0}, None
            )

        history = self.observability_state.request_history(
            model_name, active, limit=policy.window_size
        )
        stats = self._stats(model_name, active, policy, history)
        violations = self._violations(policy, stats)
        request_window_ready = len(history) >= policy.window_size
        if not violations and not request_window_ready:
            return self._result(
                model_name,
                active,
                policy,
                "insufficient_data",
                [],
                {"requests": len(history), "window_size": policy.window_size},
                None,
            )

        if not violations:
            return self._result(model_name, active, policy, "ok", [], stats, None)

        previous = self.active_state.get_previous_version(model_name)
        if not previous:
            return self._result(
                model_name,
                active,
                policy,
                "rollback_unavailable",
                violations,
                stats,
                None,
            )

        blocked_target = self.active_state.get_auto_rollback_block(model_name, previous)
        if blocked_target:
            return self._result(
                model_name,
                active,
                policy,
                "rollback_blocked",
                violations,
                stats,
                None,
                extra={"auto_rollback_block": blocked_target},
            )

        rollback = await self.lifecycle.rollback(model_name)
        self.active_state.record_auto_rollback_block(
            model_name, from_version=active, to_version=rollback["active"]
        )
        self.observability_state.clear_request_history(model_name, rollback["active"])
        await self.observability_recorder.record(
            "ROLLBACK_POLICY_TRIGGERED",
            model=model_name,
            from_version=active,
            to_version=rollback["active"],
            trigger=trigger,
            violations=violations,
        )
        return self._result(
            model_name, active, policy, "rollback_applied", violations, stats, rollback
        )

    def _stats(
        self,
        model_name: str,
        version: str,
        policy: RollbackPolicy,
        history: list[dict[str, Any]],
    ) -> RollbackStats:
        request_count = len(history)
        errors = sum(1 for row in history if row["status"] != "ok")
        compute_infer_values = [
            float(row["timings_ms"]["compute_infer"])
            for row in history
            if "compute_infer" in row.get("timings_ms", {})
        ]
        latest_resources = self.observability_state.resource_history(
            model_name, version, limit=1
        )
        latest_resource = latest_resources[-1] if latest_resources else {}
        cpu_samples = self._cpu_window_samples(model_name, version, policy)
        return RollbackStats(
            requests=request_count,
            error_rate=(
                errors / request_count if request_count >= policy.window_size else None
            ),
            compute_infer_p95_ms=(
                self._percentile(compute_infer_values, 95.0)
                if request_count >= policy.window_size
                else None
            ),
            worker_crashes=self.runtime.worker_crashes(model_name, version),
            memory_mb=latest_resource.get("memory_mb"),
            cpu_usage_percent_avg=mean(cpu_samples) if cpu_samples else None,
            cpu_usage_window_sec=(
                policy.max_cpu_usage_percent_window.window_sec
                if policy.max_cpu_usage_percent_window
                else None
            ),
            resource_samples=len(cpu_samples),
        )

    def _violations(
        self, policy: RollbackPolicy, stats: RollbackStats
    ) -> list[dict[str, Any]]:
        violations = []
        if (
            policy.max_error_rate is not None
            and stats.error_rate is not None
            and stats.error_rate > policy.max_error_rate
        ):
            violations.append(
                RollbackViolation("error_rate", stats.error_rate, policy.max_error_rate)
            )
        if (
            policy.max_compute_infer_p95_ms is not None
            and stats.compute_infer_p95_ms is not None
            and stats.compute_infer_p95_ms > policy.max_compute_infer_p95_ms
        ):
            violations.append(
                RollbackViolation(
                    "compute_infer_p95_ms",
                    stats.compute_infer_p95_ms,
                    policy.max_compute_infer_p95_ms,
                )
            )
        if (
            policy.max_worker_crashes is not None
            and stats.worker_crashes > policy.max_worker_crashes
        ):
            violations.append(
                RollbackViolation(
                    "worker_crashes", stats.worker_crashes, policy.max_worker_crashes
                )
            )
        if (
            policy.max_memory_mb is not None
            and stats.memory_mb is not None
            and stats.memory_mb > policy.max_memory_mb
        ):
            violations.append(
                RollbackViolation("memory_mb", stats.memory_mb, policy.max_memory_mb)
            )
        cpu_policy = policy.max_cpu_usage_percent_window
        if (
            cpu_policy is not None
            and stats.cpu_usage_percent_avg is not None
            and stats.cpu_usage_percent_avg > cpu_policy.max
        ):
            violations.append(
                RollbackViolation(
                    f"cpu_usage_percent_avg_{int(cpu_policy.window_sec)}s",
                    stats.cpu_usage_percent_avg,
                    cpu_policy.max,
                )
            )
        return [violation.to_dict() for violation in violations]

    def _result(
        self,
        model_name: str,
        version: str,
        policy: RollbackPolicy,
        decision: str,
        violations: list[dict[str, Any]],
        stats: dict[str, Any] | RollbackStats,
        rollback: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "model_name": model_name,
            "model_version": version,
            "policy": policy.to_dict(),
            "decision": decision,
            "violations": violations,
            "stats": stats.to_dict() if isinstance(stats, RollbackStats) else stats,
            "rollback": rollback,
        }
        if extra:
            result.update(extra)
        return result

    def _percentile(self, values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = int(round((percentile / 100.0) * (len(ordered) - 1)))
        return ordered[index]

    def _cpu_window_samples(
        self, model_name: str, version: str, policy: RollbackPolicy
    ) -> list[float]:
        cpu_policy = policy.max_cpu_usage_percent_window
        if cpu_policy is None:
            return []
        rows = self.observability_state.resource_history(
            model_name, version, window_sec=cpu_policy.window_sec
        )
        return [
            float(row["cpu_usage_percent"])
            for row in rows
            if row.get("cpu_usage_percent") is not None
        ]


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
