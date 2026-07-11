import random
from typing import Any

from t2i_eval.core.benchmark import BenchmarkSample
from t2i_eval.eval.utils.misc import get_nested_value, match_condition

DataItem = BenchmarkSample


def filter(
    data: list[DataItem],
    conditions: dict[str, Any],
) -> list[DataItem]:
    """
    Filter items by metadata conditions with nested key support.

    Examples:
    - {"category": "counting"}
    - {"metadata.include.0.class": "dog"}
    - {"metadata.include.0.count": lambda x: x >= 2}
    """
    result: list[DataItem] = []
    for item in data:
        matched = True
        for key_path, expected in conditions.items():
            value = get_nested_value(item.metadata, key_path)
            if not match_condition(value, expected):
                matched = False
                break
        if matched:
            result.append(item)
    return result


def sample(
    data: list[DataItem],
    seed: int | None,
    sample_size: int | None = None,
    group_by: str | None = None,
    sample_size_by_group: int | dict[Any, int] | None = None,
) -> list[DataItem]:
    """
    Sample items globally or per-group with nested group key support.

    - Global sample: set `sample_size` and keep `group_by=None`.
    - Grouped sample: set `group_by` and either:
      - `sample_size_by_group` as int, or
      - `sample_size_by_group` as dict[group_value, size].
    """
    if not data:
        return []

    rng = random.Random(seed)

    if group_by is None:
        shuffled = data[:]
        rng.shuffle(shuffled)
        if sample_size is None:
            return shuffled
        return shuffled[: min(sample_size, len(shuffled))]

    grouped: dict[Any, list[DataItem]] = {}
    for item in data:
        group_key = get_nested_value(item.metadata, group_by)
        grouped.setdefault(group_key, []).append(item)

    result: list[DataItem] = []
    for group_key, items in grouped.items():
        shuffled_items = items[:]
        rng.shuffle(shuffled_items)

        if isinstance(sample_size_by_group, dict):
            k = int(sample_size_by_group.get(group_key, 0))
        elif isinstance(sample_size_by_group, int):
            k = sample_size_by_group
        elif sample_size is not None:
            k = sample_size
        else:
            k = len(shuffled_items)

        if k <= 0:
            continue
        result.extend(shuffled_items[: min(k, len(shuffled_items))])

    return result
