from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests

from app.data_plane.worker_runtime.ipc import read_frame, read_message, write_message
from experiments.stage6.stage6_common import (
    apply_cpu_profile_to_execution,
    create_input_cases,
    find_free_port,
    infer_http,
    load_input_case,
    materialize_model_version,
    monitor_process,
    now_id,
    read_json,
    resource_snapshot,
    runtime_env,
    run_http_load,
    summarize_resource_samples,
    summarize,
    wait_http,
    write_json,
)


class Stage6Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path.cwd()
        raw_run_id = args.run_id or f"{now_id()}-{args.experiment}"
        self.run_id = raw_run_id.replace("\\", "_").replace("/", "_").replace(":", "_")
        self.run_dir = Path(args.output_root) / self.run_id
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact = Path(args.artifact)
        self.metadata_override = Path(args.metadata) if args.metadata else None
        self.port = find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.service: subprocess.Popen[str] | None = None
        self.service_logs: list[Any] = []

    def prepare_workspace(self, *, previous_for_active: str | None = None) -> dict[str, Any]:
        proposed_model_root = self.run_dir / "proposed" / "models"
        baseline_root = self.run_dir / "baseline"
        metadata = materialize_model_version(
            proposed_model_root,
            self.artifact,
            self.args.model,
            "v1",
            self.metadata_override,
            self.args.cpu_profile,
        )
        active_state: dict[str, Any] = {
            self.args.model: {"active": "v1", "previous": previous_for_active}
        }
        write_json(self.run_dir / "proposed" / "config" / "active_models.json", active_state)

        current = baseline_root / "current"
        current.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.artifact, current / "model.onnx")
        write_json(current / "model.json", metadata)

        backup = baseline_root / "backup-v1"
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current / "model.onnx", backup / "model.onnx")
        shutil.copy2(current / "model.json", backup / "model.json")

        input_dir = self.run_dir / "inputs"
        create_input_cases(input_dir, metadata, self.args.samples, self.args.seed)
        return {
            "metadata": metadata,
            "input_dir": input_dir,
            "proposed_model_root": proposed_model_root,
            "baseline_current": current,
            "baseline_backup": backup,
        }

    def start_service(self, *, cpu_budget: int | None = None) -> None:
        proposed = self.run_dir / "proposed"
        for name in ("config", "logs", "tmp"):
            (proposed / name).mkdir(parents=True, exist_ok=True)
        env = runtime_env(self.args.cpu_profile, os.environ)
        env.update(
            {
                "SERVING_HOST": "127.0.0.1",
                "SERVING_PORT": str(self.port),
                "SERVING_MODEL_ROOT": str(proposed / "models"),
                "SERVING_CONFIG_PATH": str(proposed / "config" / "active_models.json"),
                "SERVING_UPLOAD_TMP_ROOT": str(proposed / "tmp" / "uploads"),
                "SERVING_OBSERVABILITY_RING_PATH": str(proposed / "logs" / "observability.ring"),
                "SERVING_OBSERVABILITY_RING_MB": "32",
                "SERVING_RESOURCE_SAMPLING_INTERVAL_SEC": "0.5",
                "SERVING_CPU_BUDGET": str(cpu_budget or cpu_budget_for_profile(self.args.cpu_profile)),
            }
        )
        stdout = (proposed / "service.stdout.log").open("w", encoding="utf-8")
        stderr = (proposed / "service.stderr.log").open("w", encoding="utf-8")
        self.service_logs.extend([stdout, stderr])
        self.service = subprocess.Popen(
            [sys.executable, "-m", "app.main"],
            cwd=self.root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        ready = wait_http(f"{self.base_url}/ready", timeout_sec=self.args.startup_timeout_sec)
        if ready.get("status") != "ok":
            raise RuntimeError(f"Serving started but is not ready: {ready}")

    def stop_service(self) -> None:
        if self.service and self.service.poll() is None:
            self.service.terminate()
            try:
                self.service.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.service.kill()
                self.service.wait(timeout=10)
        for handle in self.service_logs:
            handle.close()

    def start_baseline_daemon(self, current: Path) -> subprocess.Popen[bytes]:
        log_path = current.parent / f"baseline-daemon-{time.time_ns()}.stderr.log"
        stderr = log_path.open("wb")
        self.service_logs.append(stderr)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.stage6.baseline_daemon",
                "--model-path",
                str(current / "model.onnx"),
                "--metadata",
                str(current / "model.json"),
                "--cpu-profile",
                self.args.cpu_profile,
            ],
            cwd=self.root,
            env=runtime_env(self.args.cpu_profile, os.environ),
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
        return process

    def stop_baseline_daemon(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None or process.poll() is not None:
            return
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

    def baseline_payload(
        self, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> tuple[dict[str, Any], bytes]:
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
        return {
            "command": "infer",
            "inputs": inputs,
            "outputs": [item["name"] for item in metadata["outputs"]],
        }, b"".join(payload_parts)

    def infer_baseline_daemon(
        self,
        process: subprocess.Popen[bytes],
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Baseline daemon pipes are not available")
        frame, payload = self.baseline_payload(metadata, arrays)
        started = time.perf_counter()
        write_message(process.stdin, frame, payload)
        response = read_message(process.stdout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response is None:
            raise RuntimeError("Baseline daemon exited during inference")
        response_frame, _response_payload = response
        if response_frame.get("status") != "ok":
            raise RuntimeError(str(response_frame.get("error", "Baseline inference failed")))
        timings = dict(response_frame.get("timings_ms", {}))
        timings["daemon_roundtrip"] = elapsed_ms
        return {"latency_ms": elapsed_ms, "timings_ms": timings}

    def run_baseline_daemon_load(
        self,
        process: subprocess.Popen[bytes],
        metadata: dict[str, Any],
        input_dir: Path,
        requests_count: int | None = None,
    ) -> dict[str, Any]:
        cases = sorted(input_dir.glob("input_*.npz"))
        count = requests_count or self.args.requests
        lock = threading.Lock()
        before_resources = resource_snapshot(process.pid)
        rows: list[dict[str, Any]] = []

        def call(index: int) -> dict[str, Any]:
            arrays = load_input_case(cases[index % len(cases)])
            try:
                with lock:
                    result = self.infer_baseline_daemon(process, metadata, arrays)
                return {"ok": True, **result}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, self.args.concurrency)) as executor:
            futures = [executor.submit(call, index) for index in range(count)]
            for future in futures:
                rows.append(future.result())
        wall_ms = (time.perf_counter() - started) * 1000.0
        after_resources = resource_snapshot(process.pid)
        cpu_seconds_delta = max(
            0.0,
            float(after_resources["cpu_seconds"])
            - float(before_resources["cpu_seconds"]),
        )
        wall_sec = max(wall_ms / 1000.0, 1e-12)
        ok = [row for row in rows if row["ok"]]
        return {
            "mode": "daemon",
            "ready": getattr(process, "ready_frame", {}),
            "requests": count,
            "concurrency": self.args.concurrency,
            "success": len(ok),
            "errors": [row for row in rows if not row["ok"]],
            "wall_ms": wall_ms,
            "throughput_rps": len(ok) / wall_sec,
            "latency_ms": summarize([float(row["latency_ms"]) for row in ok]),
            "timings_ms": {
                key: summarize(
                    [
                        float(row["timings_ms"][key])
                        for row in ok
                        if key in row.get("timings_ms", {})
                    ]
                )
                for key in [
                    "daemon_roundtrip",
                    "compute_input",
                    "compute_infer",
                    "compute_output",
                ]
            },
            "process_resources": {
                "rss_before_mb": before_resources["rss_mb"],
                "rss_after_mb": after_resources["rss_mb"],
                "cpu_seconds_delta": cpu_seconds_delta,
                "cpu_percent_one_core": (cpu_seconds_delta / wall_sec) * 100.0,
                "cpu_percent_n150_4_threads": (cpu_seconds_delta / (wall_sec * 4.0))
                * 100.0,
                "threads_after": after_resources["threads"],
            },
        }

    def run_baseline_daemon_once(
        self, current: Path, input_dir: Path, metadata: dict[str, Any], requests_count: int | None = None
    ) -> dict[str, Any]:
        process = self.start_baseline_daemon(current)
        try:
            return self.run_baseline_daemon_load(process, metadata, input_dir, requests_count)
        finally:
            self.stop_baseline_daemon(process)

    def baseline_cmd(
        self,
        mode: str,
        current: Path,
        input_dir: Path,
        output: Path,
        *extra: str,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "experiments.stage6.baseline_direct",
            mode,
            "--model-path",
            str(current / "model.onnx"),
            "--metadata",
            str(current / "model.json"),
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
            "--cpu-profile",
            self.args.cpu_profile,
            *extra,
        ]

    def run_baseline_json(
        self,
        current: Path,
        input_dir: Path,
        requests_count: int | None = None,
    ) -> dict[str, Any]:
        output = self.run_dir / "baseline" / f"baseline-{time.time_ns()}.json"
        command = self.baseline_cmd(
            "infer",
            current,
            input_dir,
            output,
            "--requests",
            str(requests_count or self.args.requests),
            "--concurrency",
            str(self.args.concurrency),
        )
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=runtime_env(self.args.cpu_profile, os.environ),
            text=True,
            capture_output=True,
            timeout=self.args.command_timeout_sec,
            check=True,
        )
        if not output.exists():
            return json.loads(completed.stdout)
        return read_json(output)

    def run_resources(self) -> dict[str, Any]:
        workspace = self.prepare_workspace()
        self.start_service()
        assert self.service is not None
        try:
            proposed_idle = monitor_process(self.service.pid, self.args.idle_sec)
            proposed_load = self.monitor_while(
                self.service.pid,
                lambda: run_http_load(
                    self.base_url,
                    self.args.model,
                    workspace["metadata"],
                    workspace["input_dir"],
                    self.args.requests,
                    self.args.concurrency,
                    f"{self.run_id}-proposed",
                ),
            )
        finally:
            self.stop_service()

        baseline_daemon = self.start_baseline_daemon(workspace["baseline_current"])
        try:
            baseline_idle = monitor_process(baseline_daemon.pid, self.args.idle_sec)
            baseline_load = self.monitor_while(
                baseline_daemon.pid,
                lambda: self.run_baseline_daemon_load(
                    baseline_daemon,
                    workspace["metadata"],
                    workspace["input_dir"],
                    self.args.requests,
                ),
            )
        finally:
            self.stop_baseline_daemon(baseline_daemon)
        return {
            "cpu_profile": cpu_profile_summary(self.args.cpu_profile),
            "proposed": {"idle": proposed_idle, "load": proposed_load},
            "baseline": {
                "idle": baseline_idle,
                "load": baseline_load,
            },
        }

    def run_latency(self) -> dict[str, Any]:
        workspace = self.prepare_workspace()
        self.start_service()
        try:
            proposed = run_http_load(
                self.base_url,
                self.args.model,
                workspace["metadata"],
                workspace["input_dir"],
                self.args.requests,
                self.args.concurrency,
                f"{self.run_id}-latency",
            )
        finally:
            self.stop_service()
        baseline = self.run_baseline_daemon_once(
            workspace["baseline_current"], workspace["input_dir"], workspace["metadata"]
        )
        return {
            "cpu_profile": cpu_profile_summary(self.args.cpu_profile),
            "proposed": proposed,
            "baseline": baseline,
        }

    def run_update(self) -> dict[str, Any]:
        workspace = self.prepare_workspace()
        self.start_service()
        try:
            proposed_started = time.perf_counter()
            upload_response = self.upload_version("v2", activate=False)
            upload_ms = (time.perf_counter() - proposed_started) * 1000.0
            activate_started = time.perf_counter()
            activate_response = requests.post(
                f"{self.base_url}/v1/models/{self.args.model}/versions/v2/activate",
                timeout=120,
            )
            activate_ms = (time.perf_counter() - activate_started) * 1000.0
            activate_response.raise_for_status()
            proposed_after = requests.get(f"{self.base_url}/v1/models/{self.args.model}", timeout=30).json()
        finally:
            self.stop_service()

        baseline_daemon = self.start_baseline_daemon(workspace["baseline_current"])
        baseline_started = time.perf_counter()
        try:
            self.stop_baseline_daemon(baseline_daemon)
            metadata_v2 = materialize_model_version(
                self.run_dir / "baseline-v2-inspect",
                self.artifact,
                self.args.model,
                "v2",
                self.metadata_override,
                self.args.cpu_profile,
            )
            shutil.copy2(
                self.run_dir / "baseline-v2-inspect" / self.args.model / "v2" / "model.onnx",
                workspace["baseline_current"] / "model.onnx",
            )
            write_json(workspace["baseline_current"] / "model.json", metadata_v2)
            restarted = self.start_baseline_daemon(workspace["baseline_current"])
            baseline_smoke = self.run_baseline_daemon_load(
                restarted, metadata_v2, workspace["input_dir"], requests_count=1
            )
        finally:
            self.stop_baseline_daemon(locals().get("restarted"))
        baseline_ms = (time.perf_counter() - baseline_started) * 1000.0
        return {
            "proposed": {
                "upload_ms": upload_ms,
                "activate_ms": activate_ms,
                "upload_response": upload_response,
                "activate_response": activate_response.json(),
                "model_after": proposed_after,
                "operator_command_templates": [
                    f"curl -sS -X POST http://127.0.0.1:<port>/v1/models/{self.args.model}/versions/v2/upload -H 'Content-Type: application/json' --data @upload-v2.json",
                    f"curl -sS -X POST http://127.0.0.1:<port>/v1/models/{self.args.model}/versions/v2/activate",
                ],
                "typed_command_count": 2,
            },
            "baseline": {
                "replace_and_smoke_ms": baseline_ms,
                "smoke": baseline_smoke,
                "operator_command_templates": [
                    "kill <baseline-pid>",
                    "cp ./v2/model.onnx ./current/model.onnx",
                    "cp ./v2/model.json ./current/model.json",
                    "python -m experiments.stage6.baseline_daemon --model-path ./current/model.onnx --metadata ./current/model.json --cpu-profile n150",
                    "python -m experiments.stage6.baseline_direct infer --model-path ./current/model.onnx --metadata ./current/model.json --input-dir ./inputs --requests 1 --concurrency 1 --output ./smoke.json",
                ],
                "typed_command_count": 5,
            },
        }

    def run_rollback(self) -> dict[str, Any]:
        workspace = self.prepare_workspace()
        self.start_service()
        try:
            self.upload_version("v2", activate=True)
            before = requests.get(f"{self.base_url}/v1/models/{self.args.model}", timeout=30).json()
            started = time.perf_counter()
            response = requests.post(f"{self.base_url}/v1/models/{self.args.model}/rollback", timeout=120)
            rollback_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            after = requests.get(f"{self.base_url}/v1/models/{self.args.model}", timeout=30).json()
        finally:
            self.stop_service()

        metadata_v2 = materialize_model_version(
            self.run_dir / "baseline-v2-inspect",
            self.artifact,
            self.args.model,
            "v2",
            self.metadata_override,
            self.args.cpu_profile,
        )
        shutil.copy2(
            self.run_dir / "baseline-v2-inspect" / self.args.model / "v2" / "model.onnx",
            workspace["baseline_current"] / "model.onnx",
        )
        write_json(workspace["baseline_current"] / "model.json", metadata_v2)
        baseline_daemon = self.start_baseline_daemon(workspace["baseline_current"])
        baseline_started = time.perf_counter()
        try:
            self.stop_baseline_daemon(baseline_daemon)
            shutil.copy2(workspace["baseline_backup"] / "model.onnx", workspace["baseline_current"] / "model.onnx")
            shutil.copy2(workspace["baseline_backup"] / "model.json", workspace["baseline_current"] / "model.json")
            restarted = self.start_baseline_daemon(workspace["baseline_current"])
            smoke = self.run_baseline_daemon_load(
                restarted, workspace["metadata"], workspace["input_dir"], requests_count=1
            )
        finally:
            self.stop_baseline_daemon(locals().get("restarted"))
        baseline_ms = (time.perf_counter() - baseline_started) * 1000.0
        return {
            "proposed": {
                "before": before,
                "rollback_ms": rollback_ms,
                "rollback_response": response.json(),
                "after": after,
                "operator_command_templates": [
                    f"curl -sS -X POST http://127.0.0.1:<port>/v1/models/{self.args.model}/rollback",
                ],
                "typed_command_count": 1,
            },
            "baseline": {
                "manual_restore_ms": baseline_ms,
                "smoke": smoke,
                "operator_command_templates": [
                    "ls ./backup-v1",
                    "kill <baseline-pid>",
                    "cp ./backup-v1/model.onnx ./current/model.onnx",
                    "cp ./backup-v1/model.json ./current/model.json",
                    "python -m experiments.stage6.baseline_daemon --model-path ./current/model.onnx --metadata ./current/model.json --cpu-profile n150",
                    "python -m experiments.stage6.baseline_direct infer --model-path ./current/model.onnx --metadata ./current/model.json --input-dir ./inputs --requests 1 --concurrency 1 --output ./smoke.json",
                ],
                "typed_command_count": 6,
                "note": "Rollback exists only if the operator kept a backup explicitly.",
            },
        }

    def run_recovery(self) -> dict[str, Any]:
        workspace = self.prepare_workspace()
        self.start_service()
        assert self.service is not None
        try:
            before = requests.get(f"{self.base_url}/v1/models/{self.args.model}", timeout=30).json()
            bad_started = time.perf_counter()
            bad = self.upload_bad_artifact()
            bad_ms = (time.perf_counter() - bad_started) * 1000.0
            after_bad = requests.get(f"{self.base_url}/v1/models/{self.args.model}", timeout=30).json()
            crash = self.kill_worker_and_probe(workspace["metadata"], workspace["input_dir"])
        finally:
            self.stop_service()

        bad_path = workspace["baseline_current"] / "model.onnx"
        original_bytes = bad_path.read_bytes()
        baseline_daemon = self.start_baseline_daemon(workspace["baseline_current"])
        try:
            self.stop_baseline_daemon(baseline_daemon)
            bad_path.write_bytes(b"not an onnx model")
            baseline_started = time.perf_counter()
            try:
                failed_daemon = self.start_baseline_daemon(workspace["baseline_current"])
                self.stop_baseline_daemon(failed_daemon)
                baseline_error = None
            except Exception as exc:
                baseline_error = str(exc)[-1000:]
            baseline_ms = (time.perf_counter() - baseline_started) * 1000.0
        finally:
            bad_path.write_bytes(original_bytes)
        return {
            "proposed": {
                "invalid_artifact": {
                    "before_active": before.get("active_version"),
                    "status_code": bad["status_code"],
                    "upload_ms": bad_ms,
                    "response": bad["body"],
                    "after_active": after_bad.get("active_version"),
                    "active_unchanged": before.get("active_version") == after_bad.get("active_version"),
                },
                "worker_crash": crash,
            },
            "baseline": {
                "corrupt_artifact_smoke_ms": baseline_ms,
                "error": baseline_error,
                "operator_command_templates": [
                    "ls ./backup-v1",
                    "cp ./backup-v1/model.onnx ./current/model.onnx",
                    "cp ./backup-v1/model.json ./current/model.json",
                    "python -m experiments.stage6.baseline_daemon --model-path ./current/model.onnx --metadata ./current/model.json --cpu-profile n150",
                ],
                "typed_command_count": 4,
            },
        }

    def run_diagnostics(self) -> dict[str, Any]:
        workspace = self.prepare_workspace()
        self.start_service()
        try:
            load = run_http_load(
                self.base_url,
                self.args.model,
                workspace["metadata"],
                workspace["input_dir"],
                self.args.requests,
                self.args.concurrency,
                f"{self.run_id}-diagnostics",
            )
            timings = requests.get(
                f"{self.base_url}/v1/models/{self.args.model}/timings",
                params={"limit": self.args.requests + 10},
                timeout=30,
            ).json()
            resources = requests.get(f"{self.base_url}/v1/runtime/resources", timeout=30).json()
        finally:
            self.stop_service()
        return {
            "cpu_profile": cpu_profile_summary(self.args.cpu_profile),
            "load": load,
            "timings": timings,
            "resources": resources,
        }

    def upload_version(self, version: str, *, activate: bool) -> dict[str, Any]:
        metadata = read_json(self.metadata_override) if self.metadata_override else {}
        metadata["execution"] = apply_cpu_profile_to_execution(
            metadata.get("execution", {}), self.args.cpu_profile
        )
        payload = {
            "artifact_base64": base64.b64encode(self.artifact.read_bytes()).decode("ascii"),
            "metadata": metadata,
            "activate": activate,
        }
        response = requests.post(
            f"{self.base_url}/v1/models/{self.args.model}/versions/{version}/upload",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def upload_bad_artifact(self) -> dict[str, Any]:
        payload = {
            "artifact_base64": base64.b64encode(b"not an onnx model").decode("ascii"),
            "metadata": {},
            "activate": False,
        }
        response = requests.post(
            f"{self.base_url}/v1/models/{self.args.model}/versions/bad-artifact/upload",
            json=payload,
            timeout=120,
        )
        try:
            body: Any = response.json()
        except Exception:
            body = response.text[:1000]
        return {"status_code": response.status_code, "body": body}

    def kill_worker_and_probe(self, metadata: dict[str, Any], input_dir: Path) -> dict[str, Any]:
        resources = requests.get(f"{self.base_url}/v1/runtime/resources", timeout=30).json()
        rows = resources.get("models", [])
        row = next((item for item in rows if item.get("model") == self.args.model), None)
        if not row or not row.get("pid"):
            return {"skipped": True, "reason": "worker pid is not available", "resources": resources}
        pid = int(row["pid"])
        try:
            victim = psutil.Process(pid)
            victims = [*victim.children(recursive=True), victim]
            for proc in victims:
                try:
                    proc.terminate()
                except psutil.Error:
                    pass
            _, alive = psutil.wait_procs(victims, timeout=2)
            for proc in alive:
                try:
                    proc.kill()
                except psutil.Error:
                    pass
        except psutil.Error:
            os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        input_case = next(iter(sorted(input_dir.glob("input_*.npz"))))
        arrays = load_input_case(input_case)
        attempts = []
        for index in range(2):
            try:
                latency = infer_http(
                    self.base_url,
                    self.args.model,
                    metadata,
                    arrays,
                    f"{self.run_id}-recovery-{index}",
                )
                attempts.append({"ok": True, "latency_ms": latency})
            except Exception as exc:
                attempts.append({"ok": False, "error": str(exc)})
                time.sleep(0.5)
        after = requests.get(f"{self.base_url}/v1/runtime/resources", timeout=30).json()
        after_row = next((item for item in after.get("models", []) if item.get("model") == self.args.model), None)
        recovered = any(item.get("ok") for item in attempts)
        crash_recorded = bool(after_row and int(after_row.get("worker_crashes") or 0) > 0)
        return {
            "killed_pid": pid,
            "attempts": attempts,
            "worker_after": after_row,
            "recovered_after_crash": recovered,
            "crash_recorded": crash_recorded,
            "restarted": bool(after_row and after_row.get("pid") != pid),
        }

    def monitor_while(self, pid: int, func: Any) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            samples = []
            started = time.perf_counter()
            while not future.done():
                samples.append(resource_snapshot(pid))
                time.sleep(0.25)
            samples.append(resource_snapshot(pid))
            resources = summarize_resource_samples(samples, (time.perf_counter() - started) * 1000.0)
            return {"resources": resources, "inference": future.result()}

    def monitor_process_object(self, process: subprocess.Popen[str]) -> dict[str, Any]:
        samples = []
        started = time.perf_counter()
        while process.poll() is None:
            samples.append(resource_snapshot(process.pid))
            time.sleep(0.25)
        samples.append(resource_snapshot(process.pid))
        return summarize_resource_samples(samples, (time.perf_counter() - started) * 1000.0)

    def wait_file(self, path: Path) -> None:
        deadline = time.perf_counter() + self.args.startup_timeout_sec
        while time.perf_counter() < deadline:
            if path.exists():
                return
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for {path}")


def cpu_budget_for_profile(cpu_profile: str) -> int:
    if cpu_profile == "n150":
        return 4
    return os.cpu_count() or 1


def cpu_profile_summary(cpu_profile: str) -> dict[str, Any]:
    if cpu_profile == "n150":
        return {
            "profile": "n150",
            "target_device": "Intel Processor N150 class",
            "cpu_threads": 4,
            "note": "Both baseline and serving use the same 4-thread CPU envelope.",
            "serving_cpu_budget": 4,
        }
    cpu_threads = os.cpu_count() or 1
    return {
        "profile": "host",
        "target_device": "current host",
        "cpu_threads": cpu_threads,
        "note": "Both baseline and serving use the full host CPU envelope available to this process.",
        "serving_cpu_budget": cpu_threads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 isolated comparison experiments")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "resources",
            "latency",
            "update",
            "rollback",
            "recovery",
            "diagnostics",
        ],
    )
    parser.add_argument("--model", default="yolo26n")
    parser.add_argument("--artifact", default="experiments/detection/artifacts/yolo26n.onnx")
    parser.add_argument("--metadata", default="experiments/detection/yolo26n.execution.json")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--idle-sec", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-profile", choices=["n150", "host"], default="n150")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="experiments/stage6/results")
    parser.add_argument("--startup-timeout-sec", type=float, default=90.0)
    parser.add_argument("--command-timeout-sec", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Stage6Runner(args)
    try:
        if args.experiment == "resources":
            result = runner.run_resources()
        elif args.experiment == "latency":
            result = runner.run_latency()
        elif args.experiment == "update":
            result = runner.run_update()
        elif args.experiment == "rollback":
            result = runner.run_rollback()
        elif args.experiment == "recovery":
            result = runner.run_recovery()
        elif args.experiment == "diagnostics":
            result = runner.run_diagnostics()
        else:
            raise ValueError(f"Unsupported experiment: {args.experiment}")
    finally:
        runner.stop_service()

    report = {
        "stage": 6,
        "experiment": args.experiment,
        "run_id": runner.run_id,
        "run_dir": str(runner.run_dir),
        "created_at": now_id(),
        "parameters": vars(args),
        "result": result,
    }
    report_path = runner.run_dir / f"{args.experiment}.json"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "run_dir": str(runner.run_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
