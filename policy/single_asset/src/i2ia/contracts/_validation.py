from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar


class ContractValidationError(ValueError):
    """Raised when a canonical contract value is malformed or unsupported."""


T = TypeVar("T")


def mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path} must be an object")
    return value


def required(value: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in value:
        raise ContractValidationError(f"{path}.{key} is required")
    return value[key]


def non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{path} must be a non-empty string")
    return value


def optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return non_empty_string(value, path)


def non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractValidationError(f"{path} must be a non-negative integer")
    return value


def positive_int(value: Any, path: str) -> int:
    result = non_negative_int(value, path)
    if result == 0:
        raise ContractValidationError(f"{path} must be greater than zero")
    return result


def finite_float(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractValidationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{path} must be a finite number")
    return result


def positive_float(value: Any, path: str) -> float:
    result = finite_float(value, path)
    if result <= 0:
        raise ContractValidationError(f"{path} must be greater than zero")
    return result


def string_tuple(
    value: Any,
    path: str,
    *,
    allow_empty: bool = True,
    unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{path} must be an array of strings")
    result = tuple(non_empty_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise ContractValidationError(f"{path} must not be empty")
    if unique and len(result) != len(set(result)):
        raise ContractValidationError(f"{path} must not contain duplicates")
    return result


def vector(
    value: Any,
    path: str,
    *,
    length: int | None = None,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ContractValidationError(f"{path} must be an array of finite numbers")
    result = tuple(finite_float(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise ContractValidationError(f"{path} must not be empty")
    if length is not None and len(result) != length:
        raise ContractValidationError(f"{path} must contain exactly {length} values")
    return result


def normalized_json_object(value: Any, path: str) -> dict[str, Any]:
    source = mapping(value, path)
    for key in source:
        if not isinstance(key, str):
            raise ContractValidationError(f"{path} keys must be strings")
    try:
        encoded = json.dumps(source, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{path} must contain only JSON-compatible values: {exc}") from exc
    return json.loads(encoded)


def json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, allow_nan=False))


def check_schema_version(
    value: Mapping[str, Any],
    expected: str,
    path: str,
) -> str:
    actual = non_empty_string(required(value, "schema_version", path), f"{path}.schema_version")
    if actual != expected:
        raise ContractValidationError(
            f"{path}.schema_version must be {expected!r}, got {actual!r}"
        )
    return actual


def validate_schema_version(actual: Any, expected: str, path: str) -> str:
    version = non_empty_string(actual, f"{path}.schema_version")
    if version != expected:
        raise ContractValidationError(
            f"{path}.schema_version must be {expected!r}, got {version!r}"
        )
    return version


def literal(value: Any, allowed: set[str], path: str) -> str:
    result = non_empty_string(value, path)
    if result not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ContractValidationError(f"{path} must be one of: {choices}")
    return result
