from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import msgspec

from app.common.json_codec import loads
from app.data_plane.model_runtime import ModelRuntimeAdapter, create_runtime_adapter
from app.data_plane.worker_runtime.ipc import read_message, write_frame, write_message
from app.data_plane.worker_runtime.tensor_codec import (
    decode_worker_inputs,
    encode_worker_outputs,
)
from app.schemas.model import ModelMetadata


def run_inference(
    runtime: ModelRuntimeAdapter,
    frame: dict[str, Any],
    payload: bytes,
) -> tuple[dict[str, Any], bytes]:
    input_started = time.perf_counter()
    input_feed = decode_worker_inputs(frame["inputs"], payload)
    input_ms = (time.perf_counter() - input_started) * 1000.0
    output_names = frame["outputs"]
    started = time.perf_counter()
    values = runtime.run(output_names, input_feed)
    inference_ms = (time.perf_counter() - started) * 1000.0
    output_started = time.perf_counter()
    outputs, output_payload = encode_worker_outputs(
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
    runtime = create_runtime_adapter(args.artifact, metadata)
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
            response, output_payload = run_inference(runtime, frame, payload)
            write_message(sys.stdout.buffer, response, output_payload)
        except Exception as exc:
            write_message(sys.stdout.buffer, {"status": "error", "error": str(exc)})


if __name__ == "__main__":
    main()
