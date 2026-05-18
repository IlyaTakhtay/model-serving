from __future__ import annotations

import asyncio
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import psutil

from app.common.exceptions import (
    ModelNotReadyError,
    ResourceBudgetExceededError,
    RuntimeInferenceError,
    ServingError,
    WorkerCommunicationError,
    WorkerCrashedError,
    WorkerStartupError,
)
from app.common.json_codec import dumps_text
from app.data_plane.worker_runtime.ipc import (
    read_async_frame,
    read_async_message,
    write_async_message,
)
from app.schemas.model import ModelMetadata


@dataclass
class LoadedModel:
    name: str
    version: str
    metadata: ModelMetadata
    artifact_path: Path
    process: asyncio.subprocess.Process
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    psutil_process: psutil.Process | None = None

    @property
    def pid(self) -> int:
        return int(self.process.pid)


class WorkerRuntime:
    def __init__(
        self, cpu_budget: int = 4, worker_startup_timeout_sec: float = 60.0
    ) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._worker_crashes: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._worker_restart_failures: defaultdict[tuple[str, str], int] = defaultdict(
            int
        )
        self.cpu_budget = cpu_budget
        self.worker_startup_timeout_sec = worker_startup_timeout_sec

    async def load(
        self,
        model_name: str,
        version: str,
        artifact_path: Path,
        metadata: ModelMetadata,
    ) -> dict[str, Any]:
        loaded = await self.create_session(model_name, version, artifact_path, metadata)
        return await self.replace_loaded(model_name, loaded)

    async def create_session(
        self,
        model_name: str,
        version: str,
        artifact_path: Path,
        metadata: ModelMetadata,
    ) -> LoadedModel:
        self._validate_cpu_budget(model_name, metadata)
        metadata_json = dumps_text(msgspec.to_builtins(metadata))
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "app.data_plane.worker_runtime.process",
            "--model-name",
            model_name,
            "--version",
            version,
            "--artifact",
            str(artifact_path),
            "--metadata",
            metadata_json,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            ready = await asyncio.wait_for(
                self._read_ready_frame(process), timeout=self.worker_startup_timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise WorkerStartupError(
                f"Worker startup timed out for {model_name}:{version}"
            )

        if not ready or ready.get("status") != "ready":
            process.kill()
            stderr = await self._read_stderr(process)
            raise WorkerStartupError(
                f"Worker failed to start for {model_name}:{version}: {stderr}"
            )

        psutil_process = psutil.Process(process.pid)
        psutil_process.cpu_percent(interval=None)
        return LoadedModel(
            model_name,
            version,
            metadata,
            artifact_path,
            process,
            psutil_process=psutil_process,
        )

    async def replace_loaded(
        self, model_name: str, loaded: LoadedModel
    ) -> dict[str, Any]:
        old = self._models.get(model_name)
        self._models[model_name] = loaded
        self._worker_crashes[(loaded.name, loaded.version)] = 0
        self._worker_restart_failures[(loaded.name, loaded.version)] = 0
        return await self._terminate_loaded(old)

    async def unload(self, model_name: str) -> dict[str, Any]:
        loaded = self._models.pop(model_name, None)
        return await self._terminate_loaded(loaded)

    def is_ready(self, model_name: str) -> bool:
        loaded = self._models.get(model_name)
        return bool(loaded and loaded.process.returncode is None)

    async def get_loaded(self, model_name: str) -> LoadedModel | None:
        loaded = self._models.get(model_name)
        if loaded and loaded.process.returncode is not None:
            return await self._restart_loaded(model_name, loaded)
        return loaded

    def loaded_models(self) -> dict[str, str]:
        return {
            name: loaded.version
            for name, loaded in self._models.items()
            if loaded.process.returncode is None
        }

    def process_metrics(self) -> dict[str, dict[str, float | int | str]]:
        return {
            name: self._process_metrics_for_loaded(name, loaded)
            for name, loaded in list(self._models.items())
        }

    def cpu_requested_threads(self) -> int:
        return sum(
            self._requested_threads(loaded.metadata)
            for loaded in self._models.values()
            if loaded.process.returncode is None
        )

    async def infer_tensors(
        self,
        model_name: str,
        inputs: list[dict[str, Any]],
        payload: bytes,
        output_names: list[str],
    ) -> tuple[list[dict[str, Any]], bytes, dict[str, float]]:
        loaded = await self.get_loaded(model_name)
        if not loaded:
            raise ModelNotReadyError(f"Model '{model_name}' is not loaded")
        return await self._infer_loaded(
            loaded, inputs, payload, output_names, restart_on_failure=True
        )

    async def infer_loaded(
        self,
        loaded: LoadedModel,
        inputs: list[dict[str, Any]],
        payload: bytes,
        output_names: list[str],
    ) -> tuple[list[dict[str, Any]], bytes, dict[str, float]]:
        return await self._infer_loaded(
            loaded, inputs, payload, output_names, restart_on_failure=False
        )

    async def discard_session(self, loaded: LoadedModel) -> dict[str, Any]:
        return await self._terminate_loaded(loaded)

    async def _infer_loaded(
        self,
        loaded: LoadedModel,
        inputs: list[dict[str, Any]],
        payload: bytes,
        output_names: list[str],
        *,
        restart_on_failure: bool,
    ) -> tuple[list[dict[str, Any]], bytes, dict[str, float]]:
        if loaded.process.stdin is None or loaded.process.stdout is None:
            raise WorkerCrashedError(
                f"Worker pipes are not available for model '{loaded.name}'"
            )

        request = {"command": "infer", "outputs": output_names, "inputs": inputs}
        started = time.perf_counter()
        lock_wait_started = time.perf_counter()
        async with loaded.lock:
            worker_wait_ms = (time.perf_counter() - lock_wait_started) * 1000.0
            try:
                await write_async_message(loaded.process.stdin, request, payload)
                message = await read_async_message(loaded.process.stdout)
            except Exception as exc:
                if restart_on_failure:
                    await self._handle_worker_failure(loaded.name, loaded)
                raise WorkerCommunicationError(
                    f"Worker communication failed for model '{loaded.name}': {exc}"
                ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if message is None:
            if restart_on_failure:
                await self._handle_worker_failure(loaded.name, loaded)
            raise WorkerCrashedError(
                f"Worker exited while serving model '{loaded.name}'"
            )
        response, output_payload = message
        if response.get("status") != "ok":
            raise RuntimeInferenceError(
                str(response.get("error", "Worker inference failed"))
            )
        timings = {
            "worker_wait": worker_wait_ms,
            "compute_input": float(
                response.get("timings_ms", {}).get("compute_input", 0.0)
            ),
            "compute_infer": float(
                response.get("timings_ms", {}).get("compute_infer", elapsed_ms)
            ),
            "compute_output": float(
                response.get("timings_ms", {}).get("compute_output", 0.0)
            ),
            "worker_roundtrip": elapsed_ms,
        }
        return response["outputs"], output_payload, timings

    async def _terminate_loaded(self, loaded: LoadedModel | None) -> dict[str, Any]:
        if loaded is None:
            return {"version": None, "pid": None, "status": "not_loaded"}

        memory_before = self._worker_rss_mb(loaded)
        pid = loaded.pid
        if loaded.process.returncode is None and loaded.process.stdin is not None:
            async with loaded.lock:
                try:
                    await write_async_message(loaded.process.stdin, {"command": "shutdown"})
                    await asyncio.wait_for(loaded.process.wait(), timeout=5)
                except Exception:
                    await self._force_stop_process(loaded.process)

        memory_after = self._worker_rss_mb(loaded)
        return {
            "version": loaded.version,
            "pid": pid,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "released_memory_mb": max(0.0, memory_before - memory_after),
        }

    def _worker_rss_mb(self, loaded: LoadedModel) -> float:
        if loaded.process.returncode is not None:
            return 0.0
        try:
            return self._process_tree_metrics(psutil.Process(loaded.pid))["memory_mb"]
        except psutil.Error:
            return 0.0

    def _process_cpu_seconds(self, process: psutil.Process) -> float:
        times = process.cpu_times()
        return float(times.user + times.system)

    def _process_tree_metrics(self, root: psutil.Process) -> dict[str, Any]:
        processes = [root]
        try:
            processes.extend(root.children(recursive=True))
        except psutil.Error:
            pass

        memory_mb = 0.0
        cpu_seconds = 0.0
        threads = 0
        child_pids: list[int] = []
        for process in processes:
            try:
                if process.pid != root.pid:
                    child_pids.append(process.pid)
                memory_mb += process.memory_info().rss / (1024 * 1024)
                cpu_seconds += self._process_cpu_seconds(process)
                threads += process.num_threads()
            except psutil.Error:
                continue
        return {
            "memory_mb": memory_mb,
            "cpu_seconds_total": cpu_seconds,
            "threads": threads,
            "child_pids": child_pids,
        }

    def _process_metrics_for_loaded(
        self, name: str, loaded: LoadedModel
    ) -> dict[str, float | int | str]:
        if loaded.process.returncode is not None:
            return self._stopped_process_metrics(name, loaded)
        try:
            proc = loaded.psutil_process or psutil.Process(loaded.pid)
            tree = self._process_tree_metrics(proc)
            return {
                "version": loaded.version,
                "pid": loaded.pid,
                "status": "running",
                "memory_mb": tree["memory_mb"],
                "cpu_seconds_total": tree["cpu_seconds_total"],
                "threads": tree["threads"],
                "child_pids": tree["child_pids"],
                "requested_threads": self._requested_threads(loaded.metadata),
                "worker_crashes": self.worker_crashes(name, loaded.version),
                "worker_restart_failures": self.worker_restart_failures(
                    name, loaded.version
                ),
            }
        except psutil.Error:
            return self._stopped_process_metrics(name, loaded)

    def _stopped_process_metrics(
        self, name: str, loaded: LoadedModel
    ) -> dict[str, int | str]:
        return {
            "version": loaded.version,
            "pid": loaded.pid,
            "status": "stopped",
            "worker_crashes": self.worker_crashes(name, loaded.version),
            "worker_restart_failures": self.worker_restart_failures(
                name, loaded.version
            ),
        }

    def worker_crashes(self, model_name: str, version: str | None = None) -> int:
        if version is not None:
            return int(self._worker_crashes[(model_name, version)])
        return sum(
            value
            for (name, _version), value in self._worker_crashes.items()
            if name == model_name
        )

    def worker_restart_failures(
        self, model_name: str, version: str | None = None
    ) -> int:
        if version is not None:
            return int(self._worker_restart_failures[(model_name, version)])
        return sum(
            value
            for (name, _version), value in self._worker_restart_failures.items()
            if name == model_name
        )

    async def _handle_worker_failure(
        self, model_name: str, loaded: LoadedModel
    ) -> None:
        self._record_worker_crash(loaded)
        await self._restart_loaded(model_name, loaded, count_crash=False)

    def _record_worker_crash(self, loaded: LoadedModel) -> None:
        self._worker_crashes[(loaded.name, loaded.version)] += 1

    async def _restart_loaded(
        self,
        model_name: str,
        loaded: LoadedModel,
        *,
        count_crash: bool = True,
    ) -> LoadedModel | None:
        if count_crash:
            self._record_worker_crash(loaded)
        await self._force_stop_process(loaded.process)
        try:
            restarted = await self.create_session(
                model_name, loaded.version, loaded.artifact_path, loaded.metadata
            )
        except ServingError:
            self._worker_restart_failures[(loaded.name, loaded.version)] += 1
            self._models.pop(model_name, None)
            return None
        self._models[model_name] = restarted
        return restarted

    async def _force_stop_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=2)
        except Exception:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=2)
            except Exception:
                pass

    async def _read_ready_frame(
        self, process: asyncio.subprocess.Process
    ) -> dict[str, Any] | None:
        if process.stdout is None:
            return None
        return await read_async_frame(process.stdout)

    async def _read_stderr(self, process: asyncio.subprocess.Process) -> str:
        if process.stderr is None:
            return ""
        try:
            await process.wait()
            data = await process.stderr.read()
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _validate_cpu_budget(
        self, replacing_model: str, metadata: ModelMetadata
    ) -> None:
        used = 0
        for name, loaded in self._models.items():
            if name == replacing_model:
                continue
            if loaded.process.returncode is None:
                used += self._requested_threads(loaded.metadata)
        requested = self._requested_threads(metadata)
        if used + requested > self.cpu_budget:
            raise ResourceBudgetExceededError(
                f"CPU budget exceeded: used={used}, requested={requested}, budget={self.cpu_budget}"
            )

    def _requested_threads(self, metadata: ModelMetadata) -> int:
        return max(1, int(metadata.execution.get("intra_op_num_threads", 1)))
