from __future__ import annotations

import struct
from typing import Any, BinaryIO

from app.common.json_codec import dumps_bytes, loads

HEADER = struct.Struct(">I")
MESSAGE_HEADER = struct.Struct(">IQ")


def write_frame(stream: BinaryIO, payload: dict[str, Any]) -> None:
    data = dumps_bytes(payload)
    stream.write(HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


def read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    header = stream.read(HEADER.size)
    if not header:
        return None
    if len(header) != HEADER.size:
        raise EOFError("Incomplete IPC frame header")
    size = HEADER.unpack(header)[0]
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("Incomplete IPC frame payload")
    return loads(data)


async def read_async_frame(stream: Any) -> dict[str, Any] | None:
    try:
        header = await stream.readexactly(HEADER.size)
    except EOFError:
        return None
    size = HEADER.unpack(header)[0]
    try:
        data = await stream.readexactly(size)
    except EOFError as exc:
        raise EOFError("Incomplete IPC frame payload") from exc
    return loads(data)


def write_message(
    stream: BinaryIO, header: dict[str, Any], payload: bytes = b""
) -> None:
    header_data = dumps_bytes(header)
    stream.write(MESSAGE_HEADER.pack(len(header_data), len(payload)))
    stream.write(header_data)
    stream.write(payload)
    stream.flush()


async def write_async_message(
    stream: Any, header: dict[str, Any], payload: bytes = b""
) -> None:
    header_data = dumps_bytes(header)
    stream.write(MESSAGE_HEADER.pack(len(header_data), len(payload)))
    stream.write(header_data)
    stream.write(payload)
    await stream.drain()


def read_message(stream: BinaryIO) -> tuple[dict[str, Any], bytes] | None:
    frame_header = stream.read(MESSAGE_HEADER.size)
    if not frame_header:
        return None
    if len(frame_header) != MESSAGE_HEADER.size:
        raise EOFError("Incomplete IPC message header")
    header_size, payload_size = MESSAGE_HEADER.unpack(frame_header)
    header_data = stream.read(header_size)
    if len(header_data) != header_size:
        raise EOFError("Incomplete IPC message JSON header")
    payload = stream.read(payload_size)
    if len(payload) != payload_size:
        raise EOFError("Incomplete IPC message binary payload")
    return loads(header_data), payload


async def read_async_message(stream: Any) -> tuple[dict[str, Any], bytes] | None:
    try:
        frame_header = await stream.readexactly(MESSAGE_HEADER.size)
    except EOFError:
        return None
    header_size, payload_size = MESSAGE_HEADER.unpack(frame_header)
    try:
        header_data = await stream.readexactly(header_size)
    except EOFError as exc:
        raise EOFError("Incomplete IPC message JSON header") from exc
    try:
        payload = await stream.readexactly(payload_size)
    except EOFError as exc:
        raise EOFError("Incomplete IPC message binary payload") from exc
    return loads(header_data), payload
