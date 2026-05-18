from __future__ import annotations

from typing import Any

import numpy as np

DATATYPE_TO_DTYPE = {
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "INT64": np.int64,
    "INT32": np.int32,
    "UINT8": np.uint8,
    "BOOL": np.bool_,
}

ONNX_TYPE_TO_DATATYPE = {
    "tensor(float16)": "FP16",
    "tensor(float)": "FP32",
    "tensor(double)": "FP64",
    "tensor(int64)": "INT64",
    "tensor(int32)": "INT32",
    "tensor(uint8)": "UINT8",
    "tensor(bool)": "BOOL",
}


def numpy_dtype_to_datatype(dtype: np.dtype[Any]) -> str:
    dtype = np.dtype(dtype)
    for datatype, np_dtype in DATATYPE_TO_DTYPE.items():
        if dtype == np.dtype(np_dtype):
            return datatype
    return str(dtype).upper()
