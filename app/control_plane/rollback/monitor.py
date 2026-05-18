from __future__ import annotations

from typing import Any

from app.common.exceptions import ServingError
from app.control_plane.rollback.auto_evaluator import AutoRollbackEvaluator
from app.observability.recorder import ObservabilityRecorder


class AutoRollbackMonitor:
    def __init__(
        self, evaluator: AutoRollbackEvaluator, recorder: ObservabilityRecorder
    ) -> None:
        self.evaluator = evaluator
        self.recorder = recorder

    async def on_request_recorded(self, model_name: str) -> None:
        await self._evaluate(model_name, trigger="request")

    async def on_resource_snapshot(self, model_name: str) -> None:
        await self._evaluate(model_name, trigger="resource")

    async def _evaluate(self, model_name: str, trigger: str) -> None:
        try:
            result = await self.evaluator.evaluate(model_name, trigger=trigger)
        except ServingError as exc:
            await self.recorder.record(
                "AUTO_ROLLBACK_EVALUATION_FAILED",
                model=model_name,
                trigger=trigger,
                error_code=exc.code,
                error=exc.message,
            )
        except Exception as exc:
            await self.recorder.record(
                "AUTO_ROLLBACK_EVALUATION_FAILED",
                model=model_name,
                trigger=trigger,
                error=str(exc),
            )
        else:
            await self._write_result_event(model_name, trigger, result)

    async def _write_result_event(
        self, model_name: str, trigger: str, result: dict[str, Any]
    ) -> None:
        decision = result.get("decision")
        if decision in {"ok", "disabled", "insufficient_data"}:
            return
        await self.recorder.record(
            "AUTO_ROLLBACK_EVALUATED",
            model=model_name,
            version=result.get("model_version"),
            trigger=trigger,
            decision=decision,
            violations=result.get("violations", []),
            auto_rollback_block=result.get("auto_rollback_block"),
        )
