from __future__ import annotations

from typing import Any

from msgspec import Struct, field

from app.schemas.tensor import TensorSpec


class ModelMetadata(Struct):
    name: str
    version: str
    runtime: str = "onnxruntime"
    artifact: str = "model.onnx"
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    rollback_policy: dict[str, Any] = field(default_factory=dict)
