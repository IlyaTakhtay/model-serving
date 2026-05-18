from __future__ import annotations

from typing import Any

import numpy as np

from app.common.exceptions import InvalidTensorError
from app.common.tensor_datatypes import DATATYPE_TO_DTYPE
from app.schemas.model import ModelMetadata


class TensorContractValidator:
    def validate_inputs(
        self,
        header: dict[str, Any],
        binary_payload: bytes,
        metadata: ModelMetadata,
    ) -> list[dict[str, Any]]:
        raw_inputs = header.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise InvalidTensorError(
                "Inference header must contain non-empty inputs list"
            )

        expected_inputs = {spec.name: spec for spec in metadata.inputs}
        seen_inputs: set[str] = set()
        offset = 0
        inputs: list[dict[str, Any]] = []

        for index, item in enumerate(raw_inputs):
            item = self._require_object(item, "Input", index)
            name, datatype, shape = self._parse_input_header(item, index)
            expected = expected_inputs.get(name)
            if expected is None:
                raise InvalidTensorError(
                    f"Input '{name}' is not declared by loaded model"
                )
            if datatype not in DATATYPE_TO_DTYPE:
                raise InvalidTensorError(
                    f"Unsupported binary input datatype '{datatype}'"
                )
            if datatype != expected.datatype:
                raise InvalidTensorError(
                    f"Input '{name}' expects datatype {expected.datatype}, got {datatype}"
                )
            if not self._shape_matches(expected.shape, shape):
                raise InvalidTensorError(
                    f"Input '{name}' expects shape {expected.shape}, got {shape}"
                )

            binary_size = self._binary_size(item, name, datatype, shape)
            chunk = binary_payload[offset : offset + binary_size]
            if len(chunk) != binary_size:
                raise InvalidTensorError(f"Input '{name}' binary payload is incomplete")
            offset += binary_size
            seen_inputs.add(name)
            inputs.append(
                {
                    "name": name,
                    "shape": shape,
                    "datatype": datatype,
                    "parameters": {"binary_data_size": binary_size},
                }
            )

        if offset != len(binary_payload):
            raise InvalidTensorError("Binary payload contains trailing bytes")
        missing = sorted(set(expected_inputs) - seen_inputs)
        if missing:
            raise InvalidTensorError(f"Missing model inputs: {missing}")
        return inputs

    def validate_output_names(
        self, header: dict[str, Any], metadata: ModelMetadata
    ) -> list[str]:
        raw_outputs = header.get("outputs", [])
        if not raw_outputs:
            return [spec.name for spec in metadata.outputs]
        if not isinstance(raw_outputs, list):
            raise InvalidTensorError("outputs must be a list")

        known_outputs = {spec.name for spec in metadata.outputs}
        output_names: list[str] = []
        for index, item in enumerate(raw_outputs):
            item = self._require_object(item, "Output", index)
            if "name" not in item:
                raise InvalidTensorError(f"Output #{index} is missing field 'name'")
            name = str(item["name"])
            if name not in known_outputs:
                raise InvalidTensorError(
                    f"Output '{name}' is not declared by loaded model"
                )
            output_names.append(name)
        return output_names

    def _parse_input_header(
        self, item: dict[str, Any], index: int
    ) -> tuple[str, str, list[int]]:
        try:
            name = str(item["name"])
            datatype = str(item["datatype"])
            shape = [int(dim) for dim in item["shape"]]
        except KeyError as exc:
            raise InvalidTensorError(
                f"Input #{index} is missing field '{exc.args[0]}'"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise InvalidTensorError(
                f"Input '{item.get('name', index)}' shape is invalid"
            ) from exc
        if any(dim <= 0 for dim in shape):
            raise InvalidTensorError(
                f"Input '{name}' shape must contain only positive dimensions"
            )
        return name, datatype, shape

    def _binary_size(
        self, item: dict[str, Any], name: str, datatype: str, shape: list[int]
    ) -> int:
        dtype = DATATYPE_TO_DTYPE[datatype]
        expected_size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        binary_size = int(
            item.get("parameters", {}).get("binary_data_size", expected_size)
        )
        if binary_size != expected_size:
            raise InvalidTensorError(
                f"Input '{name}' binary_data_size must be {expected_size}, got {binary_size}"
            )
        return binary_size

    def _shape_matches(self, expected: list[int], actual: list[int]) -> bool:
        if len(expected) != len(actual):
            return False
        return all(
            expected_dim == -1 or expected_dim == actual_dim
            for expected_dim, actual_dim in zip(expected, actual, strict=False)
        )

    def _require_object(self, item: Any, label: str, index: int) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise InvalidTensorError(f"{label} #{index} must be an object")
        return item
