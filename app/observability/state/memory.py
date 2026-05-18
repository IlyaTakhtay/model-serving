from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any

from app.observability.events import ObservabilityEvent, event_to_dict


class InMemoryObservabilityState:
    def __init__(
        self,
        *,
        recent_timings_maxlen: int = 1000,
        request_window_maxlen: int = 10000,
        resource_window_maxlen: int = 10000,
    ) -> None:
        self._lock = threading.Lock()
        self._recent_timings: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=recent_timings_maxlen)
        )
        self._request_windows: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=request_window_maxlen)
        )
        self._resource_windows: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=resource_window_maxlen)
        )
        self._latest_resources: dict[str, dict[str, Any]] = {}
        self._requests_total: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        self._errors_total: defaultdict[str, int] = defaultdict(int)
        self._timing_sum: defaultdict[tuple[str, str, str], float] = defaultdict(float)
        self._timing_count: defaultdict[tuple[str, str, str], int] = defaultdict(int)

    def apply(self, event: ObservabilityEvent) -> None:
        with self._lock:
            if event.event in {"INFERENCE", "INFERENCE_FAILED"}:
                self._apply_inference(event)
            elif event.event == "RESOURCE_SAMPLED":
                self._apply_resource(event)
            elif event.error_code:
                self._errors_total[event.error_code] += 1

    def recent_timings(
        self, model_name: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self._lock:
            if model_name is not None:
                rows = list(self._recent_timings.get(model_name, ()))
            else:
                rows = [
                    row
                    for model_rows in self._recent_timings.values()
                    for row in model_rows
                ]
        rows.sort(key=lambda row: float(row["ts"]))
        return rows[-limit:]

    def request_history(
        self, model_name: str, version: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 10000))
        with self._lock:
            if version is not None:
                rows = list(self._request_windows.get((model_name, version), ()))
            else:
                rows = [
                    row
                    for (model, _version), model_rows in self._request_windows.items()
                    if model == model_name
                    for row in model_rows
                ]
        rows.sort(key=lambda row: float(row["ts"]))
        return rows[-limit:]

    def clear_request_history(self, model_name: str, version: str | None = None) -> None:
        with self._lock:
            for key in list(self._request_windows):
                model, key_version = key
                if model == model_name and (version is None or key_version == version):
                    del self._request_windows[key]
            if version is None:
                self._recent_timings.pop(model_name, None)
            else:
                self._recent_timings[model_name] = deque(
                    (
                        row
                        for row in self._recent_timings.get(model_name, ())
                        if row.get("version") != version
                    ),
                    maxlen=self._recent_timings[model_name].maxlen,
                )

    def latest_resources(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {model: dict(row) for model, row in self._latest_resources.items()}

    def resource_history(
        self,
        model_name: str,
        version: str | None = None,
        *,
        window_sec: float | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 10000))
        with self._lock:
            if version is not None:
                rows = list(self._resource_windows.get((model_name, version), ()))
            else:
                rows = [
                    row
                    for (model, _version), model_rows in self._resource_windows.items()
                    if model == model_name
                    for row in model_rows
                ]
        if window_sec is not None:
            latest_ts = max((float(row["ts"]) for row in rows), default=0.0)
            cutoff = latest_ts - window_sec
            rows = [row for row in rows if float(row["ts"]) >= cutoff]
        rows.sort(key=lambda row: float(row["ts"]))
        return rows[-limit:]

    def metrics_snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                "requests_total": dict(self._requests_total),
                "errors_total": dict(self._errors_total),
                "timing_sum": dict(self._timing_sum),
                "timing_count": dict(self._timing_count),
            }

    def _apply_inference(self, event: ObservabilityEvent) -> None:
        if event.model is None or event.version is None:
            return
        status = event.status or ("error" if event.event == "INFERENCE_FAILED" else "ok")
        row = {
            "ts": event.ts,
            "request_id": event.request_id,
            "model": event.model,
            "version": event.version,
            "status": status,
            "latency_ms": event.latency_ms,
            "error_code": event.error_code,
            "timings_ms": dict(event.timings_ms or {}),
            "resources": dict(event.resources or {}),
        }
        self._request_windows[(event.model, event.version)].append(row)
        if event.timings_ms:
            self._recent_timings[event.model].append(row)
        self._requests_total[(event.model, event.version, status)] += 1
        if event.error_code:
            self._errors_total[event.error_code] += 1
        for stage, value in (event.timings_ms or {}).items():
            self._timing_sum[(event.model, event.version, stage)] += float(value)
            self._timing_count[(event.model, event.version, stage)] += 1

    def _apply_resource(self, event: ObservabilityEvent) -> None:
        if event.model is None:
            return
        data = event_to_dict(event)
        data["ts"] = event.ts
        version = event.version or str(data.get("version") or "")
        self._latest_resources[event.model] = data
        self._resource_windows[(event.model, version)].append(data)
