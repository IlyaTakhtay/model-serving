from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxruntime as ort

from app.common.tensor_datatypes import ONNX_TYPE_TO_DATATYPE
from app.schemas.model import ModelMetadata
from app.schemas.tensor import TensorSpec


class OnnxArtifactInspector:
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
