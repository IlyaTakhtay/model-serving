from __future__ import annotations

from pathlib import Path

import msgspec

from app.common.exceptions import (
    ModelNotFoundError,
    ModelStorageError,
    VersionNotFoundError,
)
from app.common.json_codec import JsonDecodeError, dumps_text, loads
from app.schemas.model import ModelMetadata


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[str]:
        try:
            return sorted(path.name for path in self.root.iterdir() if path.is_dir())
        except OSError as exc:
            raise ModelStorageError(f"Failed to list model registry: {exc}") from exc

    def list_versions(self, model_name: str) -> list[str]:
        model_dir = self.root / model_name
        if not model_dir.is_dir():
            raise ModelNotFoundError(f"Model '{model_name}' is not registered")
        try:
            return sorted(path.name for path in model_dir.iterdir() if path.is_dir())
        except OSError as exc:
            raise ModelStorageError(
                f"Failed to list versions for model '{model_name}': {exc}"
            ) from exc

    def version_dir(self, model_name: str, version: str) -> Path:
        version_dir = self.root / model_name / version
        if not version_dir.is_dir():
            raise VersionNotFoundError(
                f"Version '{version}' of model '{model_name}' is not registered"
            )
        return version_dir

    def get_metadata(self, model_name: str, version: str) -> ModelMetadata:
        path = self.version_dir(model_name, version) / "model.json"
        if not path.exists():
            raise VersionNotFoundError(
                f"Metadata file is missing for {model_name}:{version}"
            )
        try:
            return msgspec.convert(
                loads(path.read_text(encoding="utf-8")),
                ModelMetadata,
            )
        except JsonDecodeError as exc:
            raise VersionNotFoundError(
                f"Metadata JSON is malformed for {model_name}:{version}"
            ) from exc
        except OSError as exc:
            raise ModelStorageError(
                f"Failed to read metadata for {model_name}:{version}: {exc}"
            ) from exc

    def get_artifact_path(self, model_name: str, version: str) -> Path:
        metadata = self.get_metadata(model_name, version)
        path = self.version_dir(model_name, version) / metadata.artifact
        if not path.exists():
            raise VersionNotFoundError(
                f"Artifact file is missing for {model_name}:{version}"
            )
        return path

    def save_version(
        self, model_name: str, version: str, artifact: bytes, metadata: ModelMetadata
    ) -> Path:
        version_dir = self.root / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = version_dir / metadata.artifact
        try:
            artifact_path.write_bytes(artifact)
            (version_dir / "model.json").write_text(
                dumps_text(msgspec.to_builtins(metadata), pretty=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ModelStorageError(
                f"Failed to save model {model_name}:{version}: {exc}"
            ) from exc
        return artifact_path

    def save_metadata(
        self, model_name: str, version: str, metadata: ModelMetadata
    ) -> None:
        metadata_path = self.version_dir(model_name, version) / "model.json"
        try:
            metadata_path.write_text(
                dumps_text(msgspec.to_builtins(metadata), pretty=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ModelStorageError(
                f"Failed to save metadata for {model_name}:{version}: {exc}"
            ) from exc
