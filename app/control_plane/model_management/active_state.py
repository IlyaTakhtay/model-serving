from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.common.exceptions import ActiveModelStateError
from app.common.json_codec import JsonDecodeError, dumps_text, loads


class ActiveModelStateStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = loads(self.path.read_text(encoding="utf-8") or "{}")
        except OSError as exc:
            raise ActiveModelStateError(
                f"Failed to read active model state: {exc}"
            ) from exc
        except JsonDecodeError as exc:
            raise ActiveModelStateError(
                f"Active model state JSON is malformed: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ActiveModelStateError("Active model state root must be a JSON object")
        return data

    def get_active_version(self, model_name: str) -> str | None:
        return self.read().get(model_name, {}).get("active")

    def get_previous_version(self, model_name: str) -> str | None:
        return self.read().get(model_name, {}).get("previous")

    def set_active_version(self, model_name: str, version: str | None) -> str | None:
        data = self.read()
        current = data.get(model_name, {}).get("active")
        state = dict(data.get(model_name, {}))
        state.update({"active": version, "previous": current})
        data[model_name] = state
        self._write(data)
        return current

    def deactivate(self, model_name: str) -> str | None:
        data = self.read()
        current = data.get(model_name, {}).get("active")
        state = dict(data.get(model_name, {}))
        state.update({"active": None, "previous": current})
        data[model_name] = state
        self._write(data)
        return current

    def swap_active_previous(self, model_name: str) -> tuple[str | None, str | None]:
        data = self.read()
        current = data.get(model_name, {}).get("active")
        previous = data.get(model_name, {}).get("previous")
        state = dict(data.get(model_name, {}))
        state.update({"active": previous, "previous": current})
        data[model_name] = state
        self._write(data)
        return current, previous

    def record_auto_rollback_block(
        self, model_name: str, from_version: str, to_version: str
    ) -> None:
        data = self.read()
        state = dict(data.get(model_name, {}))
        state["auto_rollback_block"] = {
            "blocked_target_version": from_version,
            "blocked_after_rollback_from": from_version,
            "active_after_rollback": to_version,
            "created_at": time.time(),
        }
        data[model_name] = state
        self._write(data)

    def get_auto_rollback_block(
        self, model_name: str, target_version: str
    ) -> dict[str, Any] | None:
        block = self.read().get(model_name, {}).get("auto_rollback_block")
        if not isinstance(block, dict):
            return None
        if block.get("blocked_target_version") != target_version:
            return None
        return block

    def clear_auto_rollback_block(self, model_name: str) -> None:
        data = self.read()
        state = dict(data.get(model_name, {}))
        state.pop("auto_rollback_block", None)
        data[model_name] = state
        self._write(data)

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(dumps_text(data, pretty=True), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            raise ActiveModelStateError(
                f"Failed to write active model state: {exc}"
            ) from exc
