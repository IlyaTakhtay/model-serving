from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litestar import Request
from litestar.response import Response

from app.common.exceptions import InvalidInferenceProtocolError
from app.common.json_codec import JsonDecodeError, dumps_bytes, loads

HEADER_LENGTH = "Inference-Header-Content-Length"
MEDIA_TYPE = "application/octet-stream"


@dataclass(frozen=True)
class BinaryTensorRequest:
    header: dict[str, Any]
    payload: bytes


async def read_binary_tensor_request(request: Request) -> BinaryTensorRequest:
    body = await request.body()
    header_length = _header_length(request, len(body))
    header = _decode_header(body[:header_length])
    return BinaryTensorRequest(header=header, payload=body[header_length:])


def make_binary_tensor_response(
    header: dict[str, Any], payload: bytes
) -> Response[bytes]:
    header_bytes = dumps_bytes(header)
    return Response(
        content=header_bytes + payload,
        media_type=MEDIA_TYPE,
        headers={HEADER_LENGTH: str(len(header_bytes))},
    )


def _header_length(request: Request, body_size: int) -> int:
    try:
        header_length = int(request.headers[HEADER_LENGTH])
    except KeyError as exc:
        raise InvalidInferenceProtocolError(f"{HEADER_LENGTH} is required") from exc
    except ValueError as exc:
        raise InvalidInferenceProtocolError(
            f"{HEADER_LENGTH} must be an integer"
        ) from exc
    if header_length < 0 or header_length > body_size:
        raise InvalidInferenceProtocolError(
            f"{HEADER_LENGTH} is outside request body bounds"
        )
    return header_length


def _decode_header(data: bytes) -> dict[str, Any]:
    try:
        header = loads(data)
    except JsonDecodeError as exc:
        raise InvalidInferenceProtocolError(
            f"Inference binary header JSON is malformed: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise InvalidInferenceProtocolError(
            "Inference binary header must be a JSON object"
        )
    return header
