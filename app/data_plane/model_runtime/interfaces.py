from __future__ import annotations

from typing import Protocol

import numpy as np


class ModelRuntimeAdapter(Protocol):
    """Runtime adapter used by a worker process to execute tensor inference."""

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Run model inference for already-decoded tensors."""
