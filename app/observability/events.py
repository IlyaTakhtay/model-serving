from __future__ import annotations

import time
from typing import Any

import msgspec


class ObservabilityEvent(msgspec.Struct, omit_defaults=True):
    event: str
    ts: float
    model: str | None = None
    version: str | None = None
    request_id: str | None = None
    status: str | None = None
    error_code: str | None = None
    latency_ms: float | None = None
    timings_ms: dict[str, float] | None = None
    resources: dict[str, float] | None = None
    details: dict[str, Any] | None = None


def make_event(event: str, **fields: Any) -> ObservabilityEvent:
    ts = float(fields.pop("ts", time.time()))
    known = {
        "model": fields.pop("model", None),
        "version": fields.pop("version", fields.pop("model_version", None)),
        "request_id": fields.pop("request_id", None),
        "status": fields.pop("status", None),
        "error_code": fields.pop("error_code", None),
        "latency_ms": _optional_float(fields.pop("latency_ms", None)),
        "timings_ms": _float_dict(fields.pop("timings_ms", None)),
        "resources": _float_dict(fields.pop("resources", None)),
    }
    return ObservabilityEvent(
        event=event,
        ts=ts,
        details=fields or None,
        **known,
    )


def event_to_dict(event: ObservabilityEvent) -> dict[str, Any]:
    data = msgspec.to_builtins(event)
    # Keep the public event shape compatible with the old JSONL endpoint.
    data["ts"] = _format_ts(float(data["ts"]))
    details = data.pop("details", None)
    if details:
        data.update(details)
    return data


def _format_ts(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp)) + (
        f".{int(timestamp % 1 * 1_000_000):06d}+00:00"
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _float_dict(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    return {str(key): float(item) for key, item in dict(value).items()}
