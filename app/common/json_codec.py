from __future__ import annotations

from typing import Any

import msgspec

JsonDecodeError = msgspec.DecodeError


def loads(data: bytes | str) -> Any:
    return msgspec.json.decode(data)


def dumps_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return msgspec.json.format(msgspec.json.encode(value), indent=2)
    return msgspec.json.encode(value)


def dumps_text(value: Any, *, pretty: bool = False) -> str:
    return dumps_bytes(value, pretty=pretty).decode("utf-8")
