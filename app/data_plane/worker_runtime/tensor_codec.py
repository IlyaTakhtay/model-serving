from __future__ import annotations

from typing import Any

import numpy as np

from app.common.tensor_datatypes import DATATYPE_TO_DTYPE, numpy_dtype_to_datatype


def decode_worker_inputs(
    inputs: list[dict[str, Any]], payload: bytes
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    offset = 0
    for item in inputs:
        dtype = DATATYPE_TO_DTYPE[item["datatype"]]
        byte_size = int(item["parameters"]["binary_data_size"])
        data = payload[offset : offset + byte_size]
        offset += byte_size
        feed[item["name"]] = np.frombuffer(data, dtype=dtype).reshape(item["shape"])
    return feed


def encode_worker_outputs(
    outputs: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], bytes]:
    headers: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    for name, value in outputs.items():
        array = np.ascontiguousarray(value)
        data = array.tobytes(order="C")
        payload_parts.append(data)
        headers.append(
            {
                "name": name,
                "shape": list(array.shape),
                "datatype": numpy_dtype_to_datatype(array.dtype),
                "parameters": {"binary_data_size": len(data)},
            }
        )
    return headers, b"".join(payload_parts)
