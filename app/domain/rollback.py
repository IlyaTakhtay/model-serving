from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.common.exceptions import InvalidRequestError


@dataclass(frozen=True)
class CpuUsageWindowPolicy:
    max: float
    window_sec: float

    @classmethod
    def from_value(cls, value: Any) -> CpuUsageWindowPolicy | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise InvalidRequestError("max_cpu_usage_percent_window must be an object")
        try:
            return cls(
                max=float(value["max"]), window_sec=max(1.0, float(value["window_sec"]))
            )
        except KeyError as exc:
            raise InvalidRequestError(
                f"max_cpu_usage_percent_window.{exc.args[0]} is required"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(
                "max_cpu_usage_percent_window is invalid"
            ) from exc

    def to_dict(self) -> dict[str, float]:
        return {"max": self.max, "window_sec": self.window_sec}


@dataclass(frozen=True)
class RollbackPolicy:
    enabled: bool = False
    window_size: int = 100
    max_error_rate: float | None = 0.05
    max_compute_infer_p95_ms: float | None = 150.0
    max_worker_crashes: int | None = 0
    max_memory_mb: float | None = None
    max_cpu_usage_percent_window: CpuUsageWindowPolicy | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RollbackPolicy:
        values = dict(data or {})
        try:
            return cls(
                enabled=bool(values.get("enabled", cls.enabled)),
                window_size=max(1, int(values.get("window_size", cls.window_size))),
                max_error_rate=_optional_float(
                    values.get("max_error_rate", cls.max_error_rate)
                ),
                max_compute_infer_p95_ms=_optional_float(
                    values.get("max_compute_infer_p95_ms", cls.max_compute_infer_p95_ms)
                ),
                max_worker_crashes=_optional_int(
                    values.get("max_worker_crashes", cls.max_worker_crashes)
                ),
                max_memory_mb=_optional_float(
                    values.get("max_memory_mb", cls.max_memory_mb)
                ),
                max_cpu_usage_percent_window=CpuUsageWindowPolicy.from_value(
                    values.get(
                        "max_cpu_usage_percent_window", cls.max_cpu_usage_percent_window
                    )
                ),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"Invalid rollback_policy: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "window_size": self.window_size,
            "max_error_rate": self.max_error_rate,
            "max_compute_infer_p95_ms": self.max_compute_infer_p95_ms,
            "max_worker_crashes": self.max_worker_crashes,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_usage_percent_window": (
                self.max_cpu_usage_percent_window.to_dict()
                if self.max_cpu_usage_percent_window
                else None
            ),
        }


@dataclass(frozen=True)
class RollbackStats:
    requests: int
    error_rate: float | None
    compute_infer_p95_ms: float | None
    worker_crashes: int
    memory_mb: float | None = None
    cpu_usage_percent_avg: float | None = None
    cpu_usage_window_sec: float | None = None
    resource_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "error_rate": self.error_rate,
            "compute_infer_p95_ms": self.compute_infer_p95_ms,
            "worker_crashes": self.worker_crashes,
            "memory_mb": self.memory_mb,
            "cpu_usage_percent_avg": self.cpu_usage_percent_avg,
            "cpu_usage_window_sec": self.cpu_usage_window_sec,
            "resource_samples": self.resource_samples,
        }


@dataclass(frozen=True)
class RollbackViolation:
    metric: str
    value: float | int
    threshold: float | int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
        }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
