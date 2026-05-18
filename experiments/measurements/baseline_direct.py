from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import onnxruntime as ort
import psutil

from experiments.support.common import (
    load_input_case,
    make_session_options,
    read_json,
    runtime_env,
    summarize,
)


def make_session(model_path: Path, metadata: dict[str, Any]) -> ort.InferenceSession:
    execution = metadata.get("execution", {})
    return ort.InferenceSession(
        str(model_path),
        sess_options=make_session_options(execution),
        providers=execution.get("providers", ["CPUExecutionProvider"]),
    )


def run_inference(
    session: ort.InferenceSession,
    metadata: dict[str, Any],
    input_dir: Path,
    requests_count: int,
    concurrency: int,
) -> dict[str, Any]:
    input_paths = sorted(input_dir.glob("input_*.npz"))
    output_names = [item["name"] for item in metadata["outputs"]]
    rows: list[dict[str, Any]] = []
    proc = psutil.Process(os.getpid())
    before_times = proc.cpu_times()
    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    def call(index: int) -> dict[str, Any]:
        arrays = load_input_case(input_paths[index % len(input_paths)])
        started = time.perf_counter()
        session.run(output_names, arrays)
        return {"latency_ms": (time.perf_counter() - started) * 1000.0}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(call, index) for index in range(requests_count)]
        for future in as_completed(futures):
            rows.append(future.result())
    wall_ms = (time.perf_counter() - started) * 1000.0
    after_times = proc.cpu_times()
    rss_after_mb = proc.memory_info().rss / (1024 * 1024)
    cpu_seconds_delta = float(
        (after_times.user + after_times.system)
        - (before_times.user + before_times.system)
    )
    wall_sec = max(wall_ms / 1000.0, 1e-12)
    return {
        "requests": requests_count,
        "concurrency": concurrency,
        "success": len(rows),
        "wall_ms": wall_ms,
        "throughput_rps": len(rows) / max(wall_ms / 1000.0, 1e-12),
        "latency_ms": summarize([float(row["latency_ms"]) for row in rows]),
        "process_resources": {
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": rss_after_mb,
            "cpu_seconds_delta": cpu_seconds_delta,
            "cpu_percent_one_core": (cpu_seconds_delta / wall_sec) * 100.0,
            "cpu_percent_n150_4_threads": (cpu_seconds_delta / (wall_sec * 4.0))
            * 100.0,
            "threads_after": proc.num_threads(),
        },
    }


def write_ready(path: str | None) -> None:
    if not path:
        return
    ready_path = Path(path)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text("ready", encoding="utf-8")


def emit(data: Any, output_path: str | None) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
    print(text, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naive baseline: direct ONNX Runtime script without serving layer"
    )
    parser.add_argument("mode", choices=["hold", "infer"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cpu-profile", choices=["n150", "host"], default="n150")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(runtime_env(args.cpu_profile, os.environ))
    metadata = read_json(Path(args.metadata))
    loaded_at = time.perf_counter()
    session = make_session(Path(args.model_path), metadata)
    load_ms = (time.perf_counter() - loaded_at) * 1000.0
    write_ready(args.ready_file)

    if args.mode == "hold":
        time.sleep(args.duration_sec)
        emit({"mode": "hold", "load_ms": load_ms, "duration_sec": args.duration_sec}, args.output)
        return

    result = run_inference(
        session,
        metadata,
        Path(args.input_dir),
        args.requests,
        args.concurrency,
    )
    result["mode"] = "infer"
    result["load_ms"] = load_ms
    emit(result, args.output)


if __name__ == "__main__":
    main()
