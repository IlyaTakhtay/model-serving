from __future__ import annotations

import numpy as np

from app.schemas.model import ModelMetadata


class OpenVinoRuntimeAdapter:
    def __init__(self, artifact: str, metadata: ModelMetadata) -> None:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        core = Core()
        model = core.read_model(artifact)
        self.compiled = core.compile_model(model, "CPU", self._compile_config(metadata))
        self.request = self.compiled.create_infer_request()
        self.outputs_by_name = {
            output.get_any_name(): output for output in self.compiled.outputs
        }

    def _compile_config(self, metadata: ModelMetadata) -> dict[str, str]:
        execution = metadata.execution
        config = {
            str(key): str(value)
            for key, value in execution.get("openvino_config", {}).items()
        }
        if "intra_op_num_threads" in execution:
            config.setdefault(
                "INFERENCE_NUM_THREADS", str(execution["intra_op_num_threads"])
            )
        config.setdefault("NUM_STREAMS", str(execution.get("num_streams", 1)))
        config.setdefault(
            "PERFORMANCE_HINT", str(execution.get("performance_hint", "LATENCY"))
        )
        return config

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        result = self.request.infer(input_feed)
        values: list[np.ndarray] = []
        for name in output_names:
            output = self.outputs_by_name.get(name)
            if output is None:
                raise KeyError(f"OpenVINO output not found: {name}")
            values.append(np.asarray(result[output]))
        return values
