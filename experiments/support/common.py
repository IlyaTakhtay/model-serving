from __future__ import annotations

import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
import requests

HEADER_LENGTH = "Inference-Header-Content-Length"

DATATYPE_TO_DTYPE = {
    "BOOL": np.bool_,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
}

ONNX_TYPE_TO_DATATYPE = {
    "tensor(bool)": "BOOL",
    "tensor(uint8)": "UINT8",
    "tensor(uint16)": "UINT16",
    "tensor(uint32)": "UINT32",
    "tensor(uint64)": "UINT64",
    "tensor(int8)": "INT8",
    "tensor(int16)": "INT16",
    "tensor(int32)": "INT32",
    "tensor(int64)": "INT64",
    "tensor(float16)": "FP16",
    "tensor(float)": "FP32",
    "tensor(double)": "FP64",
}


def runtime_env(cpu_profile: str, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    cpu_threads = os.cpu_count() or 1
    if cpu_profile == "n150":
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "SERVING_CPU_BUDGET": "4",
            }
        )
    elif cpu_profile == "host":
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env.pop(key, None)
        env["SERVING_CPU_BUDGET"] = str(cpu_threads)
    else:
        raise ValueError(f"Unknown CPU profile: {cpu_profile}")
    env["ORT_DISABLE_ALL"] = "0"
    return env


def n150_env(base: dict[str, str] | None = None) -> dict[str, str]:
    return runtime_env("n150", base)


def apply_cpu_profile_to_execution(
    execution: dict[str, Any], cpu_profile: str
) -> dict[str, Any]:
    result = dict(execution)
    result.setdefault("providers", ["CPUExecutionProvider"])
    if cpu_profile == "host":
        result["intra_op_num_threads"] = os.cpu_count() or 1
        result.setdefault("inter_op_num_threads", 1)
    elif cpu_profile == "n150":
        result["intra_op_num_threads"] = 4
        result["inter_op_num_threads"] = 1
    else:
        raise ValueError(f"Unknown CPU profile: {cpu_profile}")
    return result


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": mean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def make_session_options(execution: dict[str, Any] | None) -> ort.SessionOptions:
    execution = execution or {}
    options = ort.SessionOptions()
    if "intra_op_num_threads" in execution:
        options.intra_op_num_threads = int(execution["intra_op_num_threads"])
    if "inter_op_num_threads" in execution:
        options.inter_op_num_threads = int(execution["inter_op_num_threads"])
    if "enable_cpu_mem_arena" in execution:
        options.enable_cpu_mem_arena = bool(execution["enable_cpu_mem_arena"])
    if "enable_mem_pattern" in execution:
        options.enable_mem_pattern = bool(execution["enable_mem_pattern"])
    if execution.get("execution_mode") == "parallel":
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    return options


def inspect_onnx(
    artifact_path: Path,
    model_name: str,
    version: str,
    metadata_override_path: Path | None = None,
) -> dict[str, Any]:
    override = read_json(metadata_override_path) if metadata_override_path else {}
    session = ort.InferenceSession(
        str(artifact_path),
        sess_options=make_session_options(override.get("execution")),
        providers=override.get("execution", {}).get("providers", ["CPUExecutionProvider"]),
    )
    return {
        "name": model_name,
        "version": version,
        "runtime": "onnxruntime",
        "artifact": "model.onnx",
        "inputs": [node_spec(item) for item in session.get_inputs()],
        "outputs": [node_spec(item) for item in session.get_outputs()],
        "execution": dict(override.get("execution", {})),
    }


def node_spec(node: Any) -> dict[str, Any]:
    return {
        "name": node.name,
        "datatype": ONNX_TYPE_TO_DATATYPE.get(node.type, node.type),
        "shape": [dim if isinstance(dim, int) else -1 for dim in node.shape],
    }


def materialize_model_version(
    model_root: Path,
    artifact_path: Path,
    model_name: str,
    version: str,
    metadata_override_path: Path | None = None,
    cpu_profile: str = "n150",
) -> dict[str, Any]:
    target = model_root / model_name / version
    target.mkdir(parents=True, exist_ok=True)
    artifact_bytes = artifact_path.read_bytes()
    (target / "model.onnx").write_bytes(artifact_bytes)
    metadata = inspect_onnx(artifact_path, model_name, version, metadata_override_path)
    metadata["execution"] = apply_cpu_profile_to_execution(
        metadata.get("execution", {}), cpu_profile
    )
    write_json(target / "model.json", metadata)
    return metadata


def input_shape(shape: list[int]) -> list[int]:
    return [dim if dim > 0 else 1 for dim in shape]


def create_input_cases(
    input_dir: Path,
    metadata: dict[str, Any],
    samples: int,
    seed: int = 42,
) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for sample_index in range(samples):
        arrays: dict[str, np.ndarray] = {}
        for spec in metadata["inputs"]:
            dtype = DATATYPE_TO_DTYPE[spec["datatype"]]
            shape = input_shape(spec["shape"])
            if np.issubdtype(dtype, np.floating):
                array = rng.random(shape, dtype=np.float32).astype(dtype)
            elif np.issubdtype(dtype, np.bool_):
                array = rng.integers(0, 2, size=shape).astype(dtype)
            else:
                array = rng.integers(0, 8, size=shape).astype(dtype)
            arrays[spec["name"]] = np.ascontiguousarray(array)
        path = input_dir / f"input_{sample_index:04d}.npz"
        np.savez(path, **arrays)
        paths.append(path)
    write_json(input_dir / "manifest.json", {"samples": [path.name for path in paths]})
    return paths


def list_input_cases(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("input_*.npz"))


def load_input_case(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.ascontiguousarray(data[key]) for key in data.files}


def make_binary_request(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    request_id: str,
) -> tuple[bytes, dict[str, str]]:
    inputs = []
    payload_parts = []
    for spec in metadata["inputs"]:
        array = np.ascontiguousarray(arrays[spec["name"]])
        payload = array.tobytes(order="C")
        payload_parts.append(payload)
        inputs.append(
            {
                "name": spec["name"],
                "shape": list(array.shape),
                "datatype": spec["datatype"],
                "parameters": {"binary_data_size": len(payload)},
            }
        )
    header = {
        "id": request_id,
        "inputs": inputs,
        "outputs": [{"name": item["name"]} for item in metadata["outputs"]],
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return header_bytes + b"".join(payload_parts), {
        "Content-Type": "application/octet-stream",
        HEADER_LENGTH: str(len(header_bytes)),
    }


def infer_http(
    base_url: str,
    model_name: str,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    request_id: str,
    timeout: int = 120,
    client: requests.Session | None = None,
) -> float:
    body, headers = make_binary_request(metadata, arrays, request_id)
    http_client = client or requests
    started = time.perf_counter()
    response = http_client.post(
        f"{base_url.rstrip('/')}/v1/models/{model_name}/infer",
        data=body,
        headers=headers,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    return elapsed_ms


def run_http_load(
    base_url: str,
    model_name: str,
    metadata: dict[str, Any],
    input_dir: Path,
    requests_count: int,
    concurrency: int,
    run_id: str,
) -> dict[str, Any]:
    cases = list_input_cases(input_dir)
    rows: list[dict[str, Any]] = []
    local = threading.local()

    def session() -> requests.Session:
        client = getattr(local, "client", None)
        if client is None:
            client = requests.Session()
            local.client = client
        return client

    def call(index: int) -> dict[str, Any]:
        arrays = load_input_case(cases[index % len(cases)])
        request_id = f"{run_id}-{index:06d}"
        try:
            latency_ms = infer_http(
                base_url,
                model_name,
                metadata,
                arrays,
                request_id,
                client=session(),
            )
            return {"ok": True, "latency_ms": latency_ms}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(call, index) for index in range(requests_count)]
        for future in as_completed(futures):
            rows.append(future.result())
    wall_ms = (time.perf_counter() - started) * 1000.0
    ok = [row for row in rows if row["ok"]]
    return {
        "requests": requests_count,
        "concurrency": concurrency,
        "success": len(ok),
        "errors": [row for row in rows if not row["ok"]],
        "wall_ms": wall_ms,
        "throughput_rps": len(ok) / max(wall_ms / 1000.0, 1e-12),
        "latency_ms": summarize([float(row["latency_ms"]) for row in ok]),
    }


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_http(url: str, timeout_sec: float = 60.0) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_sec
    last_error: str | None = None
    while time.perf_counter() < deadline:
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                return response.json()
            last_error = response.text[:500]
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def process_tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.Error:
        return []


def resource_snapshot(pid: int) -> dict[str, Any]:
    processes = process_tree(pid)
    rows = []
    for proc in processes:
        try:
            times = proc.cpu_times()
            rows.append(
                {
                    "pid": proc.pid,
                    "name": proc.name(),
                    "rss_mb": proc.memory_info().rss / (1024 * 1024),
                    "cpu_seconds": float(times.user + times.system),
                    "threads": proc.num_threads(),
                    "status": proc.status(),
                }
            )
        except psutil.Error:
            continue
    return {
        "processes": rows,
        "rss_mb": sum(float(row["rss_mb"]) for row in rows),
        "cpu_seconds": sum(float(row["cpu_seconds"]) for row in rows),
        "threads": sum(int(row["threads"]) for row in rows),
    }


def monitor_process(pid: int, duration_sec: float, interval_sec: float = 0.25) -> dict[str, Any]:
    samples = []
    started = time.perf_counter()
    while (time.perf_counter() - started) < duration_sec:
        samples.append(resource_snapshot(pid))
        time.sleep(interval_sec)
    return summarize_resource_samples(samples, (time.perf_counter() - started) * 1000.0)


def monitor_until_exit(process: Any, interval_sec: float = 0.25) -> dict[str, Any]:
    samples = []
    started = time.perf_counter()
    while process.poll() is None:
        samples.append(resource_snapshot(process.pid))
        time.sleep(interval_sec)
    samples.append(resource_snapshot(process.pid))
    return summarize_resource_samples(samples, (time.perf_counter() - started) * 1000.0)


def summarize_resource_samples(samples: list[dict[str, Any]], wall_ms: float) -> dict[str, Any]:
    rss_values = [float(sample["rss_mb"]) for sample in samples]
    thread_values = [float(sample["threads"]) for sample in samples]
    cpu_values = [float(sample["cpu_seconds"]) for sample in samples]
    cpu_delta = max(cpu_values[-1] - cpu_values[0], 0.0) if len(cpu_values) >= 2 else 0.0
    wall_sec = max(wall_ms / 1000.0, 1e-12)
    return {
        "samples": len(samples),
        "wall_ms": wall_ms,
        "rss_mb": summarize(rss_values),
        "threads": summarize(thread_values),
        "cpu_seconds_delta": cpu_delta,
        "cpu_percent_one_core": (cpu_delta / wall_sec) * 100.0,
        "cpu_percent_n150_4_threads": (cpu_delta / (wall_sec * 4.0)) * 100.0,
        "last_processes": samples[-1]["processes"] if samples else [],
    }
