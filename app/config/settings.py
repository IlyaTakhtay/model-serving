from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObservabilitySettings:
    ring_path: Path
    ring_size_bytes: int
    queue_size: int
    replay_records: int
    recent_timings_maxlen: int
    request_window_maxlen: int
    resource_window_maxlen: int


@dataclass(frozen=True)
class Settings:
    model_root: Path
    config_path: Path
    upload_tmp_root: Path
    host: str
    port: int
    request_max_body_size: int
    cpu_budget: int
    resource_sampling_interval_sec: float
    observability: ObservabilitySettings

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_root=Path(os.getenv("SERVING_MODEL_ROOT", "models")),
            config_path=Path(
                os.getenv("SERVING_CONFIG_PATH", "config/active_models.json")
            ),
            upload_tmp_root=Path(os.getenv("SERVING_UPLOAD_TMP_ROOT", "tmp/uploads")),
            host=os.getenv("SERVING_HOST", "127.0.0.1"),
            port=int(os.getenv("SERVING_PORT", "8080")),
            request_max_body_size=int(os.getenv("SERVING_REQUEST_MAX_BODY_MB", "64"))
            * 1024
            * 1024,
            cpu_budget=int(os.getenv("SERVING_CPU_BUDGET", "4")),
            resource_sampling_interval_sec=float(
                os.getenv("SERVING_RESOURCE_SAMPLING_INTERVAL_SEC", "5.0")
            ),
            observability=ObservabilitySettings(
                ring_path=Path(
                    os.getenv(
                        "SERVING_OBSERVABILITY_RING_PATH",
                        "logs/observability.ring",
                    )
                ),
                ring_size_bytes=int(
                    os.getenv("SERVING_OBSERVABILITY_RING_MB", "128")
                )
                * 1024
                * 1024,
                queue_size=int(
                    os.getenv("SERVING_OBSERVABILITY_QUEUE_SIZE", "10000")
                ),
                replay_records=int(
                    os.getenv("SERVING_OBSERVABILITY_REPLAY_RECORDS", "10000")
                ),
                recent_timings_maxlen=int(
                    os.getenv("SERVING_OBSERVABILITY_RECENT_TIMINGS", "1000")
                ),
                request_window_maxlen=int(
                    os.getenv("SERVING_OBSERVABILITY_REQUEST_WINDOW", "10000")
                ),
                resource_window_maxlen=int(
                    os.getenv("SERVING_OBSERVABILITY_RESOURCE_WINDOW", "10000")
                ),
            ),
        )
