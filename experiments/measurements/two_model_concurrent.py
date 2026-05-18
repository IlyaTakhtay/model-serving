from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.support.common import (
    create_input_cases,
    find_free_port,
    materialize_model_version,
    monitor_process,
    now_id,
    read_json,
    resource_snapshot,
    run_http_load,
    runtime_env,
    summarize_resource_samples,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent serving load for two active model workers"
    )
    parser.add_argument("--artifact-a", default="experiments/detection/artifacts/yolo26n.onnx")
    parser.add_argument("--metadata-a", default="experiments/detection/yolo26n.execution.json")
    parser.add_argument("--model-a", default="yolo26n")
    parser.add_argument(
        "--artifact-b",
        default="experiments/classification/artifacts/efficientnet-lite4-11.onnx",
    )
    parser.add_argument(
        "--metadata-b",
        default="experiments/classification/efficientnet-lite4.execution.json",
    )
    parser.add_argument("--model-b", default="efficientnet_lite4")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--requests-per-model", type=int, default=50)
    parser.add_argument("--concurrency-per-model", type=int, default=1)
    parser.add_argument("--threads-per-model", type=int, default=2)
    parser.add_argument("--cpu-budget", type=int, default=4)
    parser.add_argument("--cpu-profile", choices=["n150", "host"], default="n150")
    parser.add_argument("--output-root", default="experiments/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--startup-timeout-sec", type=float, default=90.0)
    parser.add_argument("--resource-interval-sec", type=float, default=0.25)
    return parser.parse_args()


def prepare_model(
    model_root: Path,
    input_root: Path,
    artifact: Path,
    metadata_override: Path,
    model_name: str,
    version: str,
    threads_per_model: int,
    samples: int,
    seed: int,
    cpu_profile: str,
) -> dict[str, Any]:
    metadata = materialize_model_version(
        model_root,
        artifact,
        model_name,
        version,
        metadata_override,
        cpu_profile,
    )
    metadata["execution"]["intra_op_num_threads"] = threads_per_model
    metadata["execution"]["inter_op_num_threads"] = 1
    write_json(model_root / model_name / version / "model.json", metadata)
    input_dir = input_root / model_name
    create_input_cases(input_dir, metadata, samples, seed)
    return {"metadata": metadata, "input_dir": input_dir}


def start_service(
    root: Path,
    run_dir: Path,
    port: int,
    cpu_budget: int,
    cpu_profile: str,
    startup_timeout_sec: float,
) -> tuple[subprocess.Popen[str], list[Any], str]:
    workspace = run_dir / "serving"
    for name in ("logs", "tmp"):
        (workspace / name).mkdir(parents=True, exist_ok=True)

    env = runtime_env(cpu_profile, os.environ)
    env.update(
        {
            "SERVING_HOST": "127.0.0.1",
            "SERVING_PORT": str(port),
            "SERVING_MODEL_ROOT": str(workspace / "models"),
            "SERVING_CONFIG_PATH": str(workspace / "config" / "active_models.json"),
            "SERVING_UPLOAD_TMP_ROOT": str(workspace / "tmp" / "uploads"),
            "SERVING_OBSERVABILITY_RING_PATH": str(workspace / "logs" / "observability.ring"),
            "SERVING_OBSERVABILITY_RING_MB": "32",
            "SERVING_RESOURCE_SAMPLING_INTERVAL_SEC": "0.5",
            "SERVING_CPU_BUDGET": str(cpu_budget),
        }
    )

    stdout = (workspace / "service.stdout.log").open("w", encoding="utf-8")
    stderr = (workspace / "service.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        cwd=root,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    wait_ready(base_url, startup_timeout_sec)
    return process, [stdout, stderr], base_url


def wait_ready(base_url: str, timeout_sec: float) -> None:
    from experiments.support.common import wait_http

    ready = wait_http(f"{base_url}/ready", timeout_sec=timeout_sec)
    if ready.get("status") != "ok":
        raise RuntimeError(f"Service is not ready: {ready}")


def stop_service(process: subprocess.Popen[str] | None, handles: list[Any]) -> None:
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    for handle in handles:
        handle.close()


def monitor_while(
    pid: int,
    interval_sec: float,
    task: Any,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        samples = []
        started = time.perf_counter()
        while not future.done():
            samples.append(resource_snapshot(pid))
            time.sleep(interval_sec)

        samples.append(resource_snapshot(pid))
        resources = summarize_resource_samples(
            samples, (time.perf_counter() - started) * 1000.0
        )
        return {"resources": resources, "load": future.result()}


def run_dual_load(
    base_url: str,
    model_a: str,
    model_b: str,
    prepared_a: dict[str, Any],
    prepared_b: dict[str, Any],
    requests_per_model: int,
    concurrency_per_model: int,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            run_http_load,
            base_url,
            model_a,
            prepared_a["metadata"],
            prepared_a["input_dir"],
            requests_per_model,
            concurrency_per_model,
            f"{run_id}-{model_a}",
        )
        future_b = executor.submit(
            run_http_load,
            base_url,
            model_b,
            prepared_b["metadata"],
            prepared_b["input_dir"],
            requests_per_model,
            concurrency_per_model,
            f"{run_id}-{model_b}",
        )
        result_a = future_a.result()
        result_b = future_b.result()
    wall_ms = (time.perf_counter() - started) * 1000.0
    total_success = int(result_a["success"]) + int(result_b["success"])
    return {
        "wall_ms": wall_ms,
        "total_requests": requests_per_model * 2,
        "total_success": total_success,
        "combined_throughput_rps": total_success / max(wall_ms / 1000.0, 1e-12),
        "models": {
            model_a: result_a,
            model_b: result_b,
        },
    }


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    artifact_a = Path(args.artifact_a)
    artifact_b = Path(args.artifact_b)
    metadata_a = Path(args.metadata_a)
    metadata_b = Path(args.metadata_b)
    if not artifact_a.exists():
        raise FileNotFoundError(
            f"Model artifact is missing: {artifact_a}. Run scripts/download_artifacts.sh"
        )
    if not artifact_b.exists():
        raise FileNotFoundError(
            f"Model artifact is missing: {artifact_b}. Run scripts/download_artifacts.sh"
        )
    run_id = (args.run_id or f"{now_id()}-two-model-concurrent").replace(":", "_")
    run_dir = Path(args.output_root) / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    model_root = run_dir / "serving" / "models"
    input_root = run_dir / "inputs"
    config_path = run_dir / "serving" / "config" / "active_models.json"

    prepared_a = prepare_model(
        model_root,
        input_root,
        artifact_a,
        metadata_a,
        args.model_a,
        args.version,
        args.threads_per_model,
        args.samples,
        seed=42,
        cpu_profile=args.cpu_profile,
    )
    prepared_b = prepare_model(
        model_root,
        input_root,
        artifact_b,
        metadata_b,
        args.model_b,
        args.version,
        args.threads_per_model,
        args.samples,
        seed=43,
        cpu_profile=args.cpu_profile,
    )
    write_json(
        config_path,
        {
            args.model_a: {"active": args.version, "previous": None},
            args.model_b: {"active": args.version, "previous": None},
        },
    )

    port = find_free_port()
    service: subprocess.Popen[str] | None = None
    handles: list[Any] = []
    try:
        service, handles, base_url = start_service(
            root,
            run_dir,
            port,
            args.cpu_budget,
            args.cpu_profile,
            args.startup_timeout_sec,
        )
        idle = monitor_process(service.pid, 1.0)
        measured = monitor_while(
            service.pid,
            args.resource_interval_sec,
            lambda: run_dual_load(
                base_url,
                args.model_a,
                args.model_b,
                prepared_a,
                prepared_b,
                args.requests_per_model,
                args.concurrency_per_model,
                run_id,
            ),
        )
        resources_after = read_json(config_path)
    finally:
        stop_service(service, handles)

    report = {
        "experiment": "two_model_concurrent",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at": now_id(),
        "parameters": vars(args),
        "active_models": resources_after,
        "result": {
            "idle_resources": idle,
            "concurrent_load": measured["load"],
            "load_resources": measured["resources"],
        },
    }
    report_path = run_dir / "two_model_concurrent.json"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
