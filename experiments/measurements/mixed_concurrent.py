from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

from app.data_plane.worker_runtime.ipc import read_frame, read_message, write_message
from experiments.measurements.two_model_concurrent import (
    prepare_model,
    start_service,
    stop_service,
)
from experiments.support.common import (
    find_free_port,
    load_input_case,
    monitor_process,
    now_id,
    read_json,
    resource_snapshot,
    runtime_env,
    summarize,
    summarize_resource_samples,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent serving and baseline load for two different ONNX models"
    )
    parser.add_argument("--artifact-a", default="experiments/detection/artifacts/yolo26n.onnx")
    parser.add_argument("--metadata-a", default="experiments/detection/yolo26n.execution.json")
    parser.add_argument("--model-a", default="yolo26n")
    parser.add_argument("--threads-a", type=int, default=3)
    parser.add_argument(
        "--artifact-b",
        default="experiments/classification/artifacts/efficientnet-lite4-11.onnx",
    )
    parser.add_argument(
        "--metadata-b",
        default="experiments/classification/efficientnet-lite4.execution.json",
    )
    parser.add_argument("--model-b", default="efficientnet_lite4")
    parser.add_argument("--threads-b", type=int, default=1)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--requests-a", type=int, default=300)
    parser.add_argument("--requests-b", type=int, default=300)
    parser.add_argument("--concurrency-a", type=int, default=1)
    parser.add_argument("--concurrency-b", type=int, default=1)
    parser.add_argument("--cpu-budget", type=int, default=4)
    parser.add_argument("--cpu-profile", choices=["n150", "host"], default="n150")
    parser.add_argument("--output-root", default="experiments/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--startup-timeout-sec", type=float, default=90.0)
    parser.add_argument("--resource-interval-sec", type=float, default=0.25)
    return parser.parse_args()


def get_json(base_url: str, path: str) -> Any:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_text(base_url: str, path: str) -> str:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=30)
    response.raise_for_status()
    return response.text


def collect_serving_observability(base_url: str, model_a: str, model_b: str) -> dict[str, Any]:
    return {
        "ready": get_json(base_url, "/ready"),
        "resources": get_json(base_url, "/v1/runtime/resources"),
        "timings": {
            model_a: get_json(base_url, f"/v1/models/{model_a}/timings?limit=1000"),
            model_b: get_json(base_url, f"/v1/models/{model_b}/timings?limit=1000"),
        },
        "events": get_json(base_url, "/events?limit=200"),
        "prometheus_metrics": get_text(base_url, "/metrics"),
    }


def run_http_load_for_model(
    base_url: str,
    model_name: str,
    metadata: dict[str, Any],
    input_dir: Path,
    requests_count: int,
    concurrency: int,
    run_id: str,
) -> dict[str, Any]:
    from experiments.support.common import run_http_load

    return run_http_load(
        base_url,
        model_name,
        metadata,
        input_dir,
        requests_count,
        concurrency,
        run_id,
    )


def run_serving_load(
    base_url: str,
    prepared_a: dict[str, Any],
    prepared_b: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            run_http_load_for_model,
            base_url,
            args.model_a,
            prepared_a["metadata"],
            prepared_a["input_dir"],
            args.requests_a,
            args.concurrency_a,
            f"{run_id}-{args.model_a}",
        )
        future_b = executor.submit(
            run_http_load_for_model,
            base_url,
            args.model_b,
            prepared_b["metadata"],
            prepared_b["input_dir"],
            args.requests_b,
            args.concurrency_b,
            f"{run_id}-{args.model_b}",
        )
        result_a = future_a.result()
        result_b = future_b.result()
    wall_ms = (time.perf_counter() - started) * 1000.0
    total_success = int(result_a["success"]) + int(result_b["success"])
    total_requests = args.requests_a + args.requests_b
    return {
        "wall_ms": wall_ms,
        "total_requests": total_requests,
        "total_success": total_success,
        "combined_throughput_rps": total_success / max(wall_ms / 1000.0, 1e-12),
        "models": {
            args.model_a: result_a,
            args.model_b: result_b,
        },
    }


def monitor_serving_while(
    base_url: str,
    pid: int,
    interval_sec: float,
    task: Any,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        external_samples = []
        internal_samples = []
        started = time.perf_counter()
        while not future.done():
            external_samples.append(resource_snapshot(pid))
            try:
                internal_samples.append(get_json(base_url, "/v1/runtime/resources"))
            except Exception as exc:
                internal_samples.append({"error": str(exc)})
            time.sleep(interval_sec)
        external_samples.append(resource_snapshot(pid))
        try:
            internal_samples.append(get_json(base_url, "/v1/runtime/resources"))
        except Exception as exc:
            internal_samples.append({"error": str(exc)})
        wall_ms = (time.perf_counter() - started) * 1000.0
        return {
            "external_resources": summarize_resource_samples(external_samples, wall_ms),
            "internal_resource_samples": internal_samples,
            "load": future.result(),
        }


def start_baseline_daemon(root: Path, model_dir: Path, cpu_profile: str) -> tuple[subprocess.Popen[bytes], Any]:
    stderr = (model_dir.parent / f"{model_dir.parent.name}-{time.time_ns()}.stderr.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "experiments.measurements.baseline_daemon",
            "--model-path",
            str(model_dir / "model.onnx"),
            "--metadata",
            str(model_dir / "model.json"),
            "--cpu-profile",
            cpu_profile,
        ],
        cwd=root,
        env=runtime_env(cpu_profile),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
    )
    if process.stdout is None:
        raise RuntimeError("Baseline daemon stdout pipe is not available")
    ready = read_frame(process.stdout)
    if not ready or ready.get("status") != "ready":
        raise RuntimeError(f"Baseline daemon failed to start: {ready}")
    process.ready_frame = ready  # type: ignore[attr-defined]
    return process, stderr


def stop_baseline_daemon(process: subprocess.Popen[bytes] | None, handle: Any | None) -> None:
    if process is not None and process.poll() is None:
        try:
            if process.stdin is not None and process.stdout is not None:
                write_message(process.stdin, {"command": "shutdown"})
                read_message(process.stdout)
            process.wait(timeout=5)
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    if handle is not None:
        handle.close()


def baseline_payload(metadata: dict[str, Any], arrays: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    inputs = []
    payload_parts = []
    for spec in metadata["inputs"]:
        array = arrays[spec["name"]]
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
    return {
        "command": "infer",
        "inputs": inputs,
        "outputs": [item["name"] for item in metadata["outputs"]],
    }, b"".join(payload_parts)


def infer_baseline(
    process: subprocess.Popen[bytes],
    lock: threading.Lock,
    metadata: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Baseline daemon pipes are not available")
    frame, payload = baseline_payload(metadata, load_input_case(input_path))
    started = time.perf_counter()
    with lock:
        write_message(process.stdin, frame, payload)
        message = read_message(process.stdout)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if message is None:
        raise RuntimeError("Baseline daemon exited during inference")
    response_frame, _response_payload = message
    if response_frame.get("status") != "ok":
        raise RuntimeError(str(response_frame.get("error", "Baseline inference failed")))
    return {
        "latency_ms": elapsed_ms,
        "timings_ms": dict(response_frame.get("timings_ms", {})),
    }


def run_baseline_load_one(
    process: subprocess.Popen[bytes],
    metadata: dict[str, Any],
    input_dir: Path,
    requests_count: int,
    concurrency: int,
) -> dict[str, Any]:
    cases = sorted(input_dir.glob("input_*.npz"))
    lock = threading.Lock()
    rows: list[dict[str, Any]] = []

    def call(index: int) -> dict[str, Any]:
        try:
            return {"ok": True, **infer_baseline(process, lock, metadata, cases[index % len(cases)])}
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
        "mode": "daemon",
        "requests": requests_count,
        "concurrency": concurrency,
        "success": len(ok),
        "errors": [row for row in rows if not row["ok"]],
        "wall_ms": wall_ms,
        "throughput_rps": len(ok) / max(wall_ms / 1000.0, 1e-12),
        "latency_ms": summarize([float(row["latency_ms"]) for row in ok]),
        "timings_ms": {
            key: summarize(
                [
                    float(row["timings_ms"][key])
                    for row in ok
                    if key in row.get("timings_ms", {})
                ]
            )
            for key in ["compute_input", "compute_infer", "compute_output"]
        },
    }


def run_baseline_load(
    process_a: subprocess.Popen[bytes],
    process_b: subprocess.Popen[bytes],
    prepared_a: dict[str, Any],
    prepared_b: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            run_baseline_load_one,
            process_a,
            prepared_a["metadata"],
            prepared_a["input_dir"],
            args.requests_a,
            args.concurrency_a,
        )
        future_b = executor.submit(
            run_baseline_load_one,
            process_b,
            prepared_b["metadata"],
            prepared_b["input_dir"],
            args.requests_b,
            args.concurrency_b,
        )
        result_a = future_a.result()
        result_b = future_b.result()
    wall_ms = (time.perf_counter() - started) * 1000.0
    total_success = int(result_a["success"]) + int(result_b["success"])
    return {
        "wall_ms": wall_ms,
        "total_requests": args.requests_a + args.requests_b,
        "total_success": total_success,
        "combined_throughput_rps": total_success / max(wall_ms / 1000.0, 1e-12),
        "models": {
            args.model_a: result_a,
            args.model_b: result_b,
        },
    }


def monitor_baseline_while(
    process_a: subprocess.Popen[bytes],
    process_b: subprocess.Popen[bytes],
    interval_sec: float,
    task: Any,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        samples_a = []
        samples_b = []
        started = time.perf_counter()
        while not future.done():
            samples_a.append(resource_snapshot(process_a.pid))
            samples_b.append(resource_snapshot(process_b.pid))
            time.sleep(interval_sec)
        samples_a.append(resource_snapshot(process_a.pid))
        samples_b.append(resource_snapshot(process_b.pid))
        wall_ms = (time.perf_counter() - started) * 1000.0
        return {
            "resources": {
                "model_a": summarize_resource_samples(samples_a, wall_ms),
                "model_b": summarize_resource_samples(samples_b, wall_ms),
            },
            "load": future.result(),
        }


def avg_metric(load: dict[str, Any], metric: str) -> float:
    models = list(load["models"].values())
    values = [float(row["latency_ms"][metric]) for row in models]
    return sum(values) / len(values)


def max_latency(load: dict[str, Any]) -> float:
    return max(float(row["latency_ms"]["max"]) for row in load["models"].values())


def summarize_load(load: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_requests": int(load["total_requests"]),
        "total_success": int(load["total_success"]),
        "wall_ms": float(load["wall_ms"]),
        "combined_throughput_rps": float(load["combined_throughput_rps"]),
        "avg_latency_ms": {
            "mean": avg_metric(load, "mean"),
            "p50": avg_metric(load, "p50"),
            "p95": avg_metric(load, "p95"),
            "max": max_latency(load),
        },
        "models": {
            model: {
                "throughput_rps": float(row["throughput_rps"]),
                "latency_ms": row["latency_ms"],
                "success": int(row["success"]),
                "requests": int(row["requests"]),
            }
            for model, row in load["models"].items()
        },
    }


def summarize_internal_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        for row in sample.get("models", []):
            model = str(row.get("model"))
            values = by_model.setdefault(model, {"cpu": [], "memory": []})
            if row.get("cpu_usage_percent") is not None:
                values["cpu"].append(float(row["cpu_usage_percent"]))
            if row.get("memory_mb") is not None:
                values["memory"].append(float(row["memory_mb"]))
    return {
        model: {
            "cpu_usage_percent": summarize(values["cpu"]),
            "memory_mb": summarize(values["memory"]),
        }
        for model, values in by_model.items()
    }


def combined_baseline_resources(resources: dict[str, Any]) -> dict[str, Any]:
    model_a = resources["model_a"]
    model_b = resources["model_b"]
    return {
        "rss_mb": {
            "mean": float(model_a["rss_mb"]["mean"]) + float(model_b["rss_mb"]["mean"]),
            "p95": float(model_a["rss_mb"]["p95"]) + float(model_b["rss_mb"]["p95"]),
            "max": float(model_a["rss_mb"]["max"]) + float(model_b["rss_mb"]["max"]),
        },
        "threads": {
            "mean": float(model_a["threads"]["mean"]) + float(model_b["threads"]["mean"]),
            "p95": float(model_a["threads"]["p95"]) + float(model_b["threads"]["p95"]),
            "max": float(model_a["threads"]["max"]) + float(model_b["threads"]["max"]),
        },
        "cpu_percent_n150_4_threads": float(model_a["cpu_percent_n150_4_threads"])
        + float(model_b["cpu_percent_n150_4_threads"]),
    }


def make_summary(report: dict[str, Any]) -> dict[str, Any]:
    result = report["result"]
    proposed = result["proposed"]
    baseline = result["baseline"]
    return {
        "experiment": report["experiment"],
        "run_id": report["run_id"],
        "scenario": "Concurrent YOLO26n + EfficientNet-Lite4 inference",
        "thread_allocation": {
            report["parameters"]["model_a"]: report["parameters"]["threads_a"],
            report["parameters"]["model_b"]: report["parameters"]["threads_b"],
            "cpu_budget": report["parameters"]["cpu_budget"],
        },
        "proposed": {
            "load": summarize_load(proposed["concurrent_load"]),
            "external_resources": proposed["load_external_resources"],
            "internal_resources": summarize_internal_samples(
                proposed["internal_resource_samples"]
            ),
        },
        "baseline": {
            "load": summarize_load(baseline["concurrent_load"]),
            "resources": baseline["load_resources"],
            "combined_resources": combined_baseline_resources(
                baseline["load_resources"]
            ),
        },
    }


def rounded(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def latency_table_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = [["Вариант", "Запросы", "mean avg, ms", "p50 avg, ms", "p95 avg, ms", "max, ms", "Combined RPS"]]
    for key, label in (("baseline", "Базовый вариант"), ("proposed", "Model-serving")):
        load = summary[key]["load"]
        lat = load["avg_latency_ms"]
        rows.append(
            [
                label,
                str(load["total_requests"]),
                rounded(lat["mean"]),
                rounded(lat["p50"]),
                rounded(lat["p95"]),
                rounded(lat["max"]),
                rounded(load["combined_throughput_rps"]),
            ]
        )
    return rows


def model_latency_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = [
        [
            "Вариант",
            "Модель",
            "Потоки",
            "Запросы",
            "mean, ms",
            "p50, ms",
            "p95, ms",
            "max, ms",
            "RPS avg по двум моделям",
        ]
    ]
    thread_allocation = summary["thread_allocation"]
    for key, label in (("baseline", "Базовый вариант"), ("proposed", "Model-serving")):
        models = summary[key]["load"]["models"]
        avg_rps = mean([float(row["throughput_rps"]) for row in models.values()])
        for model, row in models.items():
            lat = row["latency_ms"]
            rows.append(
                [
                    label,
                    model,
                    str(thread_allocation[model]),
                    str(row["requests"]),
                    rounded(lat["mean"]),
                    rounded(lat["p50"]),
                    rounded(lat["p95"]),
                    rounded(lat["max"]),
                    rounded(avg_rps),
                ]
            )
    return rows


def resources_table_rows(summary: dict[str, Any]) -> list[list[str]]:
    baseline = summary["baseline"]["combined_resources"]
    proposed = summary["proposed"]["external_resources"]
    return [
        ["Режим", "Вариант", "Память mean, MiB", "Память p95, MiB", "CPU, % от 4 потоков"],
        [
            "Нагрузка",
            "Базовый вариант",
            rounded(baseline["rss_mb"]["mean"]),
            rounded(baseline["rss_mb"]["p95"]),
            rounded(baseline["cpu_percent_n150_4_threads"]),
        ],
        [
            "Нагрузка",
            "Model-serving",
            rounded(proposed["rss_mb"]["mean"]),
            rounded(proposed["rss_mb"]["p95"]),
            rounded(proposed["cpu_percent_n150_4_threads"]),
        ],
    ]


def to_markdown(rows: list[list[str]]) -> str:
    header, *body = rows
    separator = ["---"] * len(header)
    all_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in all_rows) + "\n"


def to_csv(rows: list[list[str]]) -> str:
    return "\n".join(";".join(row) for row in rows) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def chart_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
    ]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_grouped_bars(
    path: Path,
    title: str,
    groups: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_label: str,
) -> None:
    width, height = 1280, 720
    left, right, top, bottom = 120, 70, 92, 120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = chart_font(30, True)
    label_font = chart_font(21)
    small_font = chart_font(18)

    draw.text((left, 32), title, fill=(24, 34, 48), font=title_font)
    values = [value for _, items, _ in series for value in items]
    max_value = max(values) if values else 1.0
    y_max = max_value * 1.22 if max_value > 0 else 1.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    x0, y0 = left, height - bottom

    for index in range(6):
        value = y_max * index / 5
        y = y0 - value / y_max * plot_h
        draw.line((x0, y, width - right, y), fill=(226, 231, 238), width=1)
        draw.text((26, y - 12), rounded(value, 0), fill=(86, 96, 112), font=small_font)
    draw.line((x0, top, x0, y0), fill=(88, 96, 108), width=2)
    draw.line((x0, y0, width - right, y0), fill=(88, 96, 108), width=2)
    draw.text((22, 66), y_label, fill=(72, 82, 96), font=small_font)

    group_w = plot_w / max(1, len(groups))
    bar_gap = 12
    bar_w = min(78, (group_w - 46) / max(1, len(series)) - bar_gap)
    for group_index, group in enumerate(groups):
        center = x0 + group_w * group_index + group_w / 2
        total_w = len(series) * bar_w + (len(series) - 1) * bar_gap
        start_x = center - total_w / 2
        for series_index, (_, items, color) in enumerate(series):
            value = items[group_index]
            h = value / y_max * plot_h
            bx0 = start_x + series_index * (bar_w + bar_gap)
            bx1 = bx0 + bar_w
            by0 = y0 - h
            draw.rounded_rectangle((bx0, by0, bx1, y0), radius=6, fill=color)
            text = rounded(value, 1)
            tw = draw.textlength(text, font=small_font)
            draw.text((bx0 + bar_w / 2 - tw / 2, by0 - 28), text, fill=(34, 44, 58), font=small_font)
        tw = draw.textlength(group, font=label_font)
        draw.text((center - tw / 2, y0 + 24), group, fill=(34, 44, 58), font=label_font)

    legend_x = left
    legend_y = height - 55
    for name, _, color in series:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 26, legend_y + 18), radius=4, fill=color)
        draw.text((legend_x + 36, legend_y - 4), name, fill=(34, 44, 58), font=small_font)
        legend_x += int(draw.textlength(name, font=small_font)) + 90

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_multimodel_latency_chart(path: Path, summary: dict[str, Any]) -> None:
    models = list(summary["baseline"]["load"]["models"].keys())
    baseline_values = [
        float(summary["baseline"]["load"]["models"][model]["latency_ms"]["p50"])
        for model in models
    ]
    proposed_values = [
        float(summary["proposed"]["load"]["models"][model]["latency_ms"]["p50"])
        for model in models
    ]
    draw_grouped_bars(
        path,
        "Медианная задержка в многомодельном сценарии",
        models,
        [
            ("Базовый вариант", baseline_values, (51, 102, 204)),
            ("Слой обслуживания", proposed_values, (219, 126, 54)),
        ],
        "ms",
    )


def write_experiment_outputs(run_dir: Path, report: dict[str, Any]) -> None:
    result = report["result"]
    summary = make_summary(report)
    write_json(run_dir / "mixed_concurrent.json", report)
    write_json(run_dir / "serving.json", result["proposed"])
    write_json(run_dir / "baseline.json", result["baseline"])
    write_json(run_dir / "summary.json", summary)
    write_text(run_dir / "diploma_latency_table.md", to_markdown(latency_table_rows(summary)))
    write_text(run_dir / "diploma_latency_table.csv", to_csv(latency_table_rows(summary)))
    write_text(run_dir / "diploma_model_latency_table.md", to_markdown(model_latency_rows(summary)))
    write_text(run_dir / "diploma_model_latency_table.csv", to_csv(model_latency_rows(summary)))
    write_text(run_dir / "diploma_resources_table.md", to_markdown(resources_table_rows(summary)))
    write_text(run_dir / "diploma_resources_table.csv", to_csv(resources_table_rows(summary)))
    write_multimodel_latency_chart(run_dir / "diploma_multimodel_p50.png", summary)


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    run_id = (args.run_id or f"{now_id()}-mixed-concurrent").replace(":", "_")
    run_dir = Path(args.output_root) / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    serving_model_root = run_dir / "serving" / "models"
    baseline_model_root = run_dir / "baseline" / "models"
    input_root = run_dir / "inputs"
    config_path = run_dir / "serving" / "config" / "active_models.json"

    prepared_serving_a = prepare_model(
        serving_model_root,
        input_root,
        Path(args.artifact_a),
        Path(args.metadata_a),
        args.model_a,
        args.version,
        args.threads_a,
        args.samples,
        seed=42,
        cpu_profile=args.cpu_profile,
    )
    prepared_serving_b = prepare_model(
        serving_model_root,
        input_root,
        Path(args.artifact_b),
        Path(args.metadata_b),
        args.model_b,
        args.version,
        args.threads_b,
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

    prepared_baseline_a = prepare_model(
        baseline_model_root,
        input_root,
        Path(args.artifact_a),
        Path(args.metadata_a),
        args.model_a,
        args.version,
        args.threads_a,
        args.samples,
        seed=42,
        cpu_profile=args.cpu_profile,
    )
    prepared_baseline_b = prepare_model(
        baseline_model_root,
        input_root,
        Path(args.artifact_b),
        Path(args.metadata_b),
        args.model_b,
        args.version,
        args.threads_b,
        args.samples,
        seed=43,
        cpu_profile=args.cpu_profile,
    )

    port = find_free_port()
    service: subprocess.Popen[str] | None = None
    service_handles: list[Any] = []
    try:
        service, service_handles, base_url = start_service(
            root,
            run_dir,
            port,
            args.cpu_budget,
            args.cpu_profile,
            args.startup_timeout_sec,
        )
        assert service is not None
        serving_before = collect_serving_observability(base_url, args.model_a, args.model_b)
        serving_idle_external = monitor_process(service.pid, 1.0)
        serving_measured = monitor_serving_while(
            base_url,
            service.pid,
            args.resource_interval_sec,
            lambda: run_serving_load(
                base_url,
                prepared_serving_a,
                prepared_serving_b,
                args,
                run_id,
            ),
        )
        serving_after = collect_serving_observability(base_url, args.model_a, args.model_b)
    finally:
        stop_service(service, service_handles)

    baseline_a: subprocess.Popen[bytes] | None = None
    baseline_b: subprocess.Popen[bytes] | None = None
    handle_a = None
    handle_b = None
    try:
        baseline_a, handle_a = start_baseline_daemon(
            root,
            baseline_model_root / args.model_a / args.version,
            args.cpu_profile,
        )
        baseline_b, handle_b = start_baseline_daemon(
            root,
            baseline_model_root / args.model_b / args.version,
            args.cpu_profile,
        )
        baseline_idle = {
            args.model_a: monitor_process(baseline_a.pid, 1.0),
            args.model_b: monitor_process(baseline_b.pid, 1.0),
        }
        baseline_measured = monitor_baseline_while(
            baseline_a,
            baseline_b,
            args.resource_interval_sec,
            lambda: run_baseline_load(
                baseline_a,
                baseline_b,
                prepared_baseline_a,
                prepared_baseline_b,
                args,
            ),
        )
    finally:
        stop_baseline_daemon(baseline_a, handle_a)
        stop_baseline_daemon(baseline_b, handle_b)

    report = {
        "experiment": "mixed_concurrent",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at": now_id(),
        "parameters": vars(args),
        "active_models": read_json(config_path),
        "result": {
            "proposed": {
                "before": serving_before,
                "idle_external_resources": serving_idle_external,
                "concurrent_load": serving_measured["load"],
                "load_external_resources": serving_measured["external_resources"],
                "internal_resource_samples": serving_measured["internal_resource_samples"],
                "after": serving_after,
            },
            "baseline": {
                "ready": {
                    args.model_a: getattr(baseline_a, "ready_frame", None),
                    args.model_b: getattr(baseline_b, "ready_frame", None),
                },
                "idle_resources": baseline_idle,
                "concurrent_load": baseline_measured["load"],
                "load_resources": baseline_measured["resources"],
            },
        },
    }
    write_experiment_outputs(run_dir, report)
    report_path = run_dir / "mixed_concurrent.json"
    print(json.dumps({"report": str(report_path), "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
