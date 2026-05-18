from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxruntime as ort

from app.common.tensor_datatypes import ONNX_TYPE_TO_DATATYPE
from app.control_plane.model_management.interfaces import ModelArtifactInspector
from app.schemas.model import ModelMetadata
from app.schemas.tensor import TensorSpec


class OnnxArtifactInspector:
    runtime = "onnxruntime"

    def inspect(
        self, model_name: str, version: str, artifact: str, artifact_path: Path
    ) -> ModelMetadata:
        session = ort.InferenceSession(
            str(artifact_path), providers=["CPUExecutionProvider"]
        )
        return ModelMetadata(
            name=model_name,
            version=version,
            runtime="onnxruntime",
            artifact=artifact,
            inputs=[self._spec_from_node(item) for item in session.get_inputs()],
            outputs=[self._spec_from_node(item) for item in session.get_outputs()],
            execution={},
        )

    def _spec_from_node(self, node: Any) -> TensorSpec:
        return TensorSpec(
            name=node.name,
            datatype=ONNX_TYPE_TO_DATATYPE.get(node.type, node.type),
            shape=[dim if isinstance(dim, int) else -1 for dim in node.shape],
        )


class ArtifactInspectorRegistry:
    def __init__(self, inspectors: list[ModelArtifactInspector]) -> None:
        self._inspectors = {
            inspector.runtime.lower(): inspector for inspector in inspectors
        }

    def inspect(
        self,
        runtime: str,
        model_name: str,
        version: str,
        artifact: str,
        artifact_path: Path,
    ) -> ModelMetadata:
        inspector = self._inspectors.get(runtime.lower())
        if inspector is None:
            supported = ", ".join(sorted(self._inspectors))
            raise ValueError(
                f"Unsupported artifact runtime '{runtime}'. Supported: {supported}"
            )
        return inspector.inspect(model_name, version, artifact, artifact_path)
