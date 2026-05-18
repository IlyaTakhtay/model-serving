from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import msgspec
import numpy as np
import onnxruntime as ort

from app.common.json_codec import loads
from app.common.tensor_datatypes import DATATYPE_TO_DTYPE, numpy_dtype_to_datatype
from app.data_plane.worker_runtime.ipc import read_message, write_frame, write_message
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


class OnnxRuntimeExecutor:
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


class OpenVinoExecutor:
    def __init__(self, artifact: str, metadata: ModelMetadata) -> None:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        core = Core()
        model = core.read_model(artifact)
        config = self._compile_config(metadata)
        self.compiled = core.compile_model(model, "CPU", config)
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


def create_executor(artifact: str, metadata: ModelMetadata) -> Any:
    runtime = metadata.runtime.lower()
    if runtime in ("onnxruntime", "ort"):
        return OnnxRuntimeExecutor(artifact, metadata)
    if runtime in ("openvino", "ov"):
        return OpenVinoExecutor(artifact, metadata)
    raise ValueError(f"Unsupported model runtime: {metadata.runtime}")


def decode_inputs(
    inputs: list[dict[str, Any]], payload: bytes
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    offset = 0
    for item in inputs:
        dtype = DATATYPE_TO_DTYPE[item["datatype"]]
        byte_size = int(item["parameters"]["binary_data_size"])
        data = payload[offset : offset + byte_size]
        offset += byte_size
        feed[item["name"]] = np.frombuffer(data, dtype=dtype).reshape(item["shape"])
    return feed


def encode_outputs(
    outputs: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], bytes]:
    headers: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    for name, value in outputs.items():
        array = np.ascontiguousarray(value)
        data = array.tobytes(order="C")
        payload_parts.append(data)
        headers.append(
            {
                "name": name,
                "shape": list(array.shape),
                "datatype": numpy_dtype_to_datatype(array.dtype),
                "parameters": {"binary_data_size": len(data)},
            }
        )
    return headers, b"".join(payload_parts)


def run_inference(
    executor: Any, frame: dict[str, Any], payload: bytes
) -> tuple[dict[str, Any], bytes]:
    input_started = time.perf_counter()
    input_feed = decode_inputs(frame["inputs"], payload)
    input_ms = (time.perf_counter() - input_started) * 1000.0
    output_names = frame["outputs"]
    started = time.perf_counter()
    values = executor.run(output_names, input_feed)
    inference_ms = (time.perf_counter() - started) * 1000.0
    output_started = time.perf_counter()
    outputs, output_payload = encode_outputs(
        dict(zip(output_names, values, strict=True))
    )
    output_ms = (time.perf_counter() - output_started) * 1000.0
    return {
        "status": "ok",
        "inference_ms": inference_ms,
        "timings_ms": {
            "compute_input": input_ms,
            "compute_infer": inference_ms,
            "compute_output": output_ms,
        },
        "outputs": outputs,
    }, output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Model worker process")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    metadata = msgspec.convert(loads(args.metadata), ModelMetadata)
    executor = create_executor(args.artifact, metadata)
    write_frame(
        sys.stdout.buffer,
        {"status": "ready", "model": args.model_name, "version": args.version},
    )

    while True:
        message = read_message(sys.stdin.buffer)
        if message is None:
            return
        frame, payload = message
        command = frame.get("command")
        if command == "shutdown":
            write_message(sys.stdout.buffer, {"status": "ok"})
            return
        if command != "infer":
            write_message(
                sys.stdout.buffer,
                {"status": "error", "error": f"Unknown command: {command}"},
            )
            continue
        try:
            response, output_payload = run_inference(executor, frame, payload)
            write_message(sys.stdout.buffer, response, output_payload)
        except Exception as exc:
            write_message(sys.stdout.buffer, {"status": "error", "error": str(exc)})


if __name__ == "__main__":
    main()
