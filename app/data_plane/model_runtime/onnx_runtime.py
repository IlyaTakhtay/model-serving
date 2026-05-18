from __future__ import annotations

import numpy as np
import onnxruntime as ort

from app.schemas.model import ModelMetadata

INT_SESSION_OPTIONS = {
    "intra_op_num_threads": "intra_op_num_threads",
    "inter_op_num_threads": "inter_op_num_threads",
}
BOOL_SESSION_OPTIONS = {
    "enable_cpu_mem_arena": "enable_cpu_mem_arena",
    "enable_mem_pattern": "enable_mem_pattern",
}


def make_session_options(metadata: ModelMetadata) -> ort.SessionOptions:
    execution = metadata.execution
    options = ort.SessionOptions()
    for key, attr in INT_SESSION_OPTIONS.items():
        if key in execution:
            setattr(options, attr, int(execution[key]))
    for key, attr in BOOL_SESSION_OPTIONS.items():
        if key in execution:
            setattr(options, attr, bool(execution[key]))
    if execution.get("execution_mode") == "parallel":
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    return options


class OnnxRuntimeAdapter:
    def __init__(self, artifact: str, metadata: ModelMetadata) -> None:
        self.session = ort.InferenceSession(
            artifact,
            sess_options=make_session_options(metadata),
            providers=metadata.execution.get("providers", ["CPUExecutionProvider"]),
        )

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        return self.session.run(output_names, input_feed)
