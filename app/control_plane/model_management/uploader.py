from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import msgspec

from app.common.exceptions import InvalidUploadError
from app.control_plane.model_management.artifact_inspector import OnnxArtifactInspector
from app.control_plane.model_management.lifecycle import ModelLifecycle
from app.control_plane.model_management.registry import ModelRegistry
from app.observability.recorder import ObservabilityRecorder
from app.schemas.model import ModelMetadata
from app.schemas.upload import ModelUploadChunkRequest, ModelUploadRequest


class ModelUploader:
    def __init__(
        self,
        registry: ModelRegistry,
        lifecycle: ModelLifecycle,
        artifact_inspector: OnnxArtifactInspector,
        upload_tmp_root: Path,
        recorder: ObservabilityRecorder,
    ) -> None:
        self.registry = registry
        self.lifecycle = lifecycle
        self.artifact_inspector = artifact_inspector
        self.upload_tmp_root = upload_tmp_root
        self.recorder = recorder
        self.upload_tmp_root.mkdir(parents=True, exist_ok=True)

    async def upload(
        self, model_name: str, version: str, request: ModelUploadRequest
    ) -> dict[str, Any]:
        artifact = self._decode_base64_artifact(request.artifact_base64)
        return await self._save_uploaded_artifact(
            model_name, version, artifact, request.metadata, request.activate
        )

    async def upload_chunk(
        self, model_name: str, version: str, request: ModelUploadChunkRequest
    ) -> dict[str, Any]:
        self._validate_chunk_request(request)
        chunk = self._decode_base64_artifact(request.artifact_base64)
        upload_dir = self._upload_dir(request.upload_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / f"{request.chunk_index:08d}.part").write_bytes(chunk)

        received = len(list(upload_dir.glob("*.part")))
        await self.recorder.record(
            "MODEL_UPLOAD_CHUNK_RECEIVED",
            model=model_name,
            version=version,
            upload_id=request.upload_id,
            chunk_index=request.chunk_index,
            total_chunks=request.total_chunks,
            received_chunks=received,
        )

        if received < request.total_chunks:
            return {
                "model_name": model_name,
                "model_version": version,
                "upload_id": request.upload_id,
                "status": "partial",
                "received_chunks": received,
                "total_chunks": request.total_chunks,
            }

        try:
            artifact = self._assemble_upload(upload_dir, request.total_chunks)
            response = await self._save_uploaded_artifact(
                model_name, version, artifact, request.metadata, request.activate
            )
            response["upload_id"] = request.upload_id
            response["status"] = "completed"
            return response
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    async def _save_uploaded_artifact(
        self,
        model_name: str,
        version: str,
        artifact: bytes,
        request_metadata: dict[str, Any] | None,
        activate: bool,
    ) -> dict[str, Any]:
        artifact_name = (
            request_metadata.get("artifact", "model.onnx")
            if request_metadata
            else "model.onnx"
        )
        metadata = self._metadata_from_uploaded_artifact(
            model_name, version, artifact_name, artifact
        )
        self._apply_metadata_overrides(metadata, request_metadata)
        self.registry.save_version(model_name, version, artifact, metadata)
        await self.recorder.record("MODEL_UPLOADED", model=model_name, version=version)

        response = {
            "model_name": model_name,
            "model_version": version,
            "metadata": msgspec.to_builtins(metadata),
        }
        if activate:
            response["activation"] = await self.lifecycle.activate(model_name, version)
        return response

    def _apply_metadata_overrides(
        self, metadata: ModelMetadata, request_metadata: dict[str, Any] | None
    ) -> None:
        if not request_metadata:
            return
        metadata.runtime = request_metadata.get("runtime", metadata.runtime)
        metadata.execution = dict(request_metadata.get("execution", metadata.execution))
        metadata.rollback_policy = dict(
            request_metadata.get("rollback_policy", metadata.rollback_policy)
        )

    def _metadata_from_uploaded_artifact(
        self,
        model_name: str,
        version: str,
        artifact_name: str,
        artifact: bytes,
    ) -> ModelMetadata:
        with NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp.write(artifact)
            tmp_path = Path(tmp.name)
        try:
            return self.artifact_inspector.inspect(
                model_name, version, artifact_name, tmp_path
            )
        except Exception as exc:
            raise InvalidUploadError(
                f"Uploaded artifact is not a valid ONNX model: {exc}"
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    def _validate_chunk_request(self, request: ModelUploadChunkRequest) -> None:
        if request.total_chunks <= 0:
            raise InvalidUploadError("total_chunks must be greater than zero")
        if request.chunk_index < 0 or request.chunk_index >= request.total_chunks:
            raise InvalidUploadError("chunk_index must be in range [0, total_chunks)")

    def _assemble_upload(self, upload_dir: Path, total_chunks: int) -> bytes:
        missing = [
            index
            for index in range(total_chunks)
            if not (upload_dir / f"{index:08d}.part").exists()
        ]
        if missing:
            raise InvalidUploadError(f"Missing chunks: {missing}")
        return b"".join(
            (upload_dir / f"{index:08d}.part").read_bytes()
            for index in range(total_chunks)
        )

    def _decode_base64_artifact(self, artifact_base64: str) -> bytes:
        try:
            return base64.b64decode(artifact_base64, validate=True)
        except Exception as exc:
            raise InvalidUploadError("artifact_base64 is not valid base64") from exc

    def _upload_dir(self, upload_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", upload_id)
        if not safe_id:
            raise InvalidUploadError(
                "upload_id must contain at least one safe character"
            )
        return self.upload_tmp_root / safe_id
