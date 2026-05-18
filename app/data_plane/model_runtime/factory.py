from __future__ import annotations

from app.data_plane.model_runtime.interfaces import ModelRuntimeAdapter
from app.data_plane.model_runtime.onnx_runtime import OnnxRuntimeAdapter
from app.data_plane.model_runtime.openvino_runtime import OpenVinoRuntimeAdapter
from app.schemas.model import ModelMetadata


def create_runtime_adapter(
    artifact: str, metadata: ModelMetadata
) -> ModelRuntimeAdapter:
    runtime = metadata.runtime.lower()
    if runtime in ("onnxruntime", "ort"):
        return OnnxRuntimeAdapter(artifact, metadata)
    if runtime in ("openvino", "ov"):
        return OpenVinoRuntimeAdapter(artifact, metadata)
    raise ValueError(f"Unsupported model runtime: {metadata.runtime}")
