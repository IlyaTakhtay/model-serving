from __future__ import annotations

from typing import Any

from msgspec import Struct


class ModelUploadRequest(Struct, omit_defaults=True):
    artifact_base64: str
    metadata: dict[str, Any] | None = None
    activate: bool = False


class ModelUploadChunkRequest(Struct, omit_defaults=True):
    upload_id: str
    chunk_index: int
    total_chunks: int
    artifact_base64: str
    metadata: dict[str, Any] | None = None
    activate: bool = False
