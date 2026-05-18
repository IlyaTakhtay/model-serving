from __future__ import annotations
from msgspec import Struct
class TensorSpec(Struct):
    name: str
    datatype: str
    shape: list[int]
