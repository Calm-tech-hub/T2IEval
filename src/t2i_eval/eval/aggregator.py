from typing import Any

from t2i_eval.eval.utils.misc import (
    Reduction,
    aggregate_numbers,
    get_nested_value,
    set_nested_value,
)


def aggregate_metric(
    metadatas: list[dict[str, Any]],
    value_key: str,
    output_key: str,
    reduction: Reduction = Reduction.MEAN,
    group_by: str | list[str] | None = None,
    final_reduction: Reduction | None = None,
    default: float = 0.0,
) -> dict[str, Any]:
    """Aggregate metrics with simple float values under each sample metadata."""

    if group_by is None:
        values = [get_nested_value(metadata, value_key) for metadata in metadatas]
        return {
            output_key: aggregate_numbers(values, reduction=reduction, default=default)
        }

    grouped_values: dict[tuple[Any, ...], list[Any]] = {}
    for metadata in metadatas:
        if isinstance(group_by, str):
            group_key = (get_nested_value(metadata, group_by),)
        else:
            if len(group_by) == 0:
                raise ValueError("`group_by` list must not be empty.")
            group_key = tuple(get_nested_value(metadata, key) for key in group_by)
        value = get_nested_value(metadata, value_key)
        grouped_values.setdefault(group_key, []).append(value)

    flat_grouped_metric: dict[tuple[Any, ...], float] = {
        group_key: aggregate_numbers(values, reduction=reduction, default=default)
        for group_key, values in grouped_values.items()
    }
    grouped_metric: dict[Any, Any] = {}
    for keys, value in flat_grouped_metric.items():
        set_nested_value(grouped_metric, [str(part) for part in keys], value)

    if final_reduction is not None:
        return {
            output_key: aggregate_numbers(
                flat_grouped_metric.values(),
                reduction=final_reduction,
                default=default,
            )
        }

    return {output_key: grouped_metric}


def aggregate_metric_from_list(
    metadatas: list[dict[str, Any]],
    list_key: str,
    value_key: str,
    output_key: str,
    reduction: Reduction = Reduction.MEAN,
    group_by: str | list[str] | None = None,
    final_reduction: Reduction | None = None,
    default: float = 0.0,
) -> dict[str, Any]:
    """
    First flattens the specified list of metrics from all samples, then applies aggregation. This is useful when each sample contains a list of metrics (e.g., question-level scores), and we want to compute an overall metric across all items in the lists.

    Example metadata shape:
      {
        "metrics_list": [
          {"property": "attribute", "score": 1.0},
          {"property": "relation", "score": 0.0},
        ]
      }
    """
    if group_by is None:
        values: list[Any] = []
        for metadata in metadatas:
            items = get_nested_value(metadata, list_key, default=[])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                values.append(get_nested_value(item, value_key))
        return {
            output_key: aggregate_numbers(values, reduction=reduction, default=default)
        }

    grouped_values: dict[tuple[Any, ...], list[Any]] = {}
    for metadata in metadatas:
        items = get_nested_value(metadata, list_key, default=[])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(group_by, str):
                group_key = (get_nested_value(item, group_by),)
            else:
                if len(group_by) == 0:
                    raise ValueError("`group_by` list must not be empty.")
                group_key = tuple(get_nested_value(item, key) for key in group_by)
            value = get_nested_value(item, value_key)
            grouped_values.setdefault(group_key, []).append(value)

    flat_grouped_metric: dict[tuple[Any, ...], float] = {
        group_key: aggregate_numbers(values, reduction=reduction, default=default)
        for group_key, values in grouped_values.items()
    }
    grouped_metric: dict[Any, Any] = {}
    for keys, value in flat_grouped_metric.items():
        set_nested_value(grouped_metric, [str(part) for part in keys], value)

    if final_reduction is not None:
        return {
            output_key: aggregate_numbers(
                flat_grouped_metric.values(),
                reduction=final_reduction,
                default=default,
            )
        }

    return {output_key: grouped_metric}
