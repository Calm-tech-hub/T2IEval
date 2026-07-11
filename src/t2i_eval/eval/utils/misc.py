import base64
import io
from collections.abc import Iterable
from enum import Enum
from typing import Any

import torch


class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"


def pil_to_base64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def aggregate_numbers(
    values: Iterable[Any],
    reduction: Reduction = Reduction.MEAN,
    default: float = 0.0,
) -> float:
    numbers = [float(v) for v in values if isinstance(v, int | float | bool)]
    if not numbers:
        return default

    if reduction == Reduction.MEAN:
        return float(sum(numbers) / len(numbers))
    if reduction == Reduction.SUM:
        return float(sum(numbers))
    if reduction == Reduction.MAX:
        return float(max(numbers))
    if reduction == Reduction.MIN:
        return float(min(numbers))

    raise ValueError(f"Unsupported reduction: {reduction}")


def get_nested_value(data: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested value from dict by dot path, e.g. "a.b.0.c"."""
    current: Any = data
    for key in key_path.split("."):
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue

        if isinstance(current, list):
            try:
                index = int(key)
            except ValueError:
                return default
            if index < 0 or index >= len(current):
                return default
            current = current[index]
            continue

        return default
    return current


def set_nested_value(data: dict[str, Any], keys: list[str], value: Any) -> None:
    current: dict[str, Any] = data
    for key in keys[:-1]:
        nested = current.get(key)
        if not isinstance(nested, dict):
            nested = {}
            current[key] = nested
        current = nested
    current[keys[-1]] = value


def device_to_str(device: str | torch.device | int) -> str:
    """Normalize common device representations to a string value."""
    if isinstance(device, str):
        return device
    elif isinstance(device, torch.device):
        return str(device)
    elif isinstance(device, int):
        if device < 0:
            return "cpu"
        return f"cuda:{device}"
    else:
        raise ValueError(f"Unsupported device type: {type(device)}")


def match_condition(value: Any, expected: Any) -> bool:
    if callable(expected):
        return bool(expected(value))
    if isinstance(expected, set | list | tuple):
        return value in expected
    return value == expected
