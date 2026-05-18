from __future__ import annotations

import psutil


class PrometheusMetricsExporter:
    def render(
        self,
        metrics_snapshot: dict,
        loaded_models: dict[str, str],
        model_processes: dict[str, dict] | None = None,
        cpu_budget: int | None = None,
        cpu_requested: int | None = None,
    ) -> str:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_times = process.cpu_times()
        cpu_seconds_total = float(cpu_times.user + cpu_times.system)
        lines = [
            "# TYPE serving_requests_total counter",
        ]
        for (model, version, status), value in sorted(
            metrics_snapshot["requests_total"].items()
        ):
            lines.append(
                f'serving_requests_total{{model="{model}",version="{version}",status="{status}"}} {value}'
            )
        lines.append("# TYPE serving_errors_total counter")
        for code, value in sorted(metrics_snapshot["errors_total"].items()):
            lines.append(f'serving_errors_total{{code="{code}"}} {value}')
        lines.append("# TYPE serving_inference_timing_ms_sum counter")
        for (model, version, stage), value in sorted(metrics_snapshot["timing_sum"].items()):
            lines.append(
                f'serving_inference_timing_ms_sum{{model="{model}",version="{version}",stage="{stage}"}} {value:.6f}'
            )
        lines.append("# TYPE serving_inference_timing_ms_count counter")
        for (model, version, stage), value in sorted(metrics_snapshot["timing_count"].items()):
            lines.append(
                f'serving_inference_timing_ms_count{{model="{model}",version="{version}",stage="{stage}"}} {value}'
            )
        if model_processes:
            lines.append("# TYPE serving_model_worker_crashes_total counter")
            for model, data in sorted(model_processes.items()):
                version = data.get("version")
                crashes = int(data.get("worker_crashes", 0))
                if version:
                    lines.append(
                        f'serving_model_worker_crashes_total{{model="{model}",version="{version}"}} {crashes}'
                    )
        lines.append("# TYPE serving_model_loaded gauge")
        for model, version in sorted(loaded_models.items()):
            lines.append(
                f'serving_model_loaded{{model="{model}",version="{version}"}} 1'
            )
        if model_processes:
            self._append_worker_metrics(lines, model_processes)
        if cpu_budget is not None:
            lines.append("# TYPE serving_cpu_budget_threads gauge")
            lines.append(f"serving_cpu_budget_threads {int(cpu_budget)}")
        if cpu_requested is not None:
            lines.append("# TYPE serving_cpu_requested_threads gauge")
            lines.append(f"serving_cpu_requested_threads {int(cpu_requested)}")
        lines.append("# TYPE serving_process_memory_mb gauge")
        lines.append(f"serving_process_memory_mb {memory_mb:.6f}")
        lines.append("# TYPE serving_process_cpu_seconds_total counter")
        lines.append(f"serving_process_cpu_seconds_total {cpu_seconds_total:.6f}")
        return "\n".join(lines) + "\n"

    def _append_worker_metrics(
        self, lines: list[str], model_processes: dict[str, dict]
    ) -> None:
        lines.append("# TYPE serving_model_process_memory_mb gauge")
        for model, data in sorted(model_processes.items()):
            if data.get("status") == "running":
                lines.append(
                    f'serving_model_process_memory_mb{{model="{model}",version="{data["version"]}",pid="{data["pid"]}"}} {float(data["memory_mb"]):.6f}'
                )
        lines.append("# TYPE serving_model_process_cpu_seconds_total counter")
        for model, data in sorted(model_processes.items()):
            if data.get("status") == "running":
                lines.append(
                    f'serving_model_process_cpu_seconds_total{{model="{model}",version="{data["version"]}",pid="{data["pid"]}"}} {float(data["cpu_seconds_total"]):.6f}'
                )
        lines.append("# TYPE serving_model_process_threads gauge")
        for model, data in sorted(model_processes.items()):
            if data.get("status") == "running":
                lines.append(
                    f'serving_model_process_threads{{model="{model}",version="{data["version"]}",pid="{data["pid"]}"}} {int(data["threads"])}'
                )
        lines.append("# TYPE serving_model_process_cpu_usage_percent gauge")
        for model, data in sorted(model_processes.items()):
            if (
                data.get("status") == "running"
                and data.get("cpu_usage_percent") is not None
            ):
                lines.append(
                    f'serving_model_process_cpu_usage_percent{{model="{model}",version="{data["version"]}",pid="{data["pid"]}"}} {float(data["cpu_usage_percent"]):.6f}'
                )
        lines.append("# TYPE serving_model_requested_threads gauge")
        for model, data in sorted(model_processes.items()):
            if data.get("status") == "running":
                lines.append(
                    f'serving_model_requested_threads{{model="{model}",version="{data["version"]}",pid="{data["pid"]}"}} {int(data["requested_threads"])}'
                )
