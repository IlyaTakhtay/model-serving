from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.data_plane.worker_runtime.ipc import read_message, write_frame, write_message
from experiments.stage6.stage6_common import (
    DATATYPE_TO_DTYPE,
    make_session_options,
    read_json,
    runtime_env,
)


def make_session(model_path: Path, metadata: dict[str, Any]) -> ort.InferenceSession:
    execution = metadata.get("execution", {})
    return ort.InferenceSession(
        str(model_path),
        sess_options=make_session_options(execution),
        providers=execution.get("providers", ["CPUExecutionProvider"]),
    )


def decode_inputs(inputs: list[dict[str, Any]], payload: bytes) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    offset = 0
    for item in inputs:
        dtype = DATATYPE_TO_DTYPE[item["datatype"]]
        byte_size = int(item["parameters"]["binary_data_size"])
        chunk = payload[offset : offset + byte_size]
        offset += byte_size
        feed[item["name"]] = np.frombuffer(chunk, dtype=dtype).reshape(item["shape"])
    return feed


def datatype_from_numpy(dtype: np.dtype[Any]) -> str:
    dtype = np.dtype(dtype)
    for datatype, np_dtype in DATATYPE_TO_DTYPE.items():
        if dtype == np.dtype(np_dtype):
            return datatype
    return str(dtype).upper()


def encode_outputs(
    names: list[str], values: list[np.ndarray]
) -> tuple[list[dict[str, Any]], bytes]:
    headers = []
    payload_parts = []
    for name, value in zip(names, values, strict=True):
        array = np.ascontiguousarray(value)
        payload = array.tobytes(order="C")
        payload_parts.append(payload)
        headers.append(
            {
                "name": name,
                "shape": list(array.shape),
                "datatype": datatype_from_numpy(array.dtype),
                "parameters": {"binary_data_size": len(payload)},
            }
        )
    return headers, b"".join(payload_parts)


def run_inference(
    session: ort.InferenceSession, frame: dict[str, Any], payload: bytes
) -> tuple[dict[str, Any], bytes]:
    input_started = time.perf_counter()
    feed = decode_inputs(frame["inputs"], payload)
    input_ms = (time.perf_counter() - input_started) * 1000.0
    output_names = list(frame["outputs"])
    infer_started = time.perf_counter()
    values = session.run(output_names, feed)
    infer_ms = (time.perf_counter() - infer_started) * 1000.0
    output_started = time.perf_counter()
    output_headers, output_payload = encode_outputs(output_names, values)
    output_ms = (time.perf_counter() - output_started) * 1000.0
    return {
        "status": "ok",
        "outputs": output_headers,
        "timings_ms": {
            "compute_input": input_ms,
            "compute_infer": infer_ms,
            "compute_output": output_ms,
        },
    }, output_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naive long-running ONNX Runtime baseline process"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--cpu-profile", choices=["n150", "host"], default="n150")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(runtime_env(args.cpu_profile, os.environ))
    metadata = read_json(Path(args.metadata))
    started = time.perf_counter()
    session = make_session(Path(args.model_path), metadata)
    load_ms = (time.perf_counter() - started) * 1000.0
    write_frame(
        sys.stdout.buffer,
        {
            "status": "ready",
            "mode": "naive-baseline-daemon",
            "model": metadata.get("name"),
            "version": metadata.get("version"),
            "load_ms": load_ms,
        },
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
            response, response_payload = run_inference(session, frame, payload)
            write_message(sys.stdout.buffer, response, response_payload)
        except Exception as exc:
            write_message(sys.stdout.buffer, {"status": "error", "error": str(exc)})


if __name__ == "__main__":
    main()
