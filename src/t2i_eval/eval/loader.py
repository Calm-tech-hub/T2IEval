import json
from collections.abc import Callable
from typing import Any

from datasets import Dataset, load_dataset

from t2i_eval.core.schema import GenerationConfig

RowToEvalItems = Callable[
    [dict[str, Any], GenerationConfig],
    list[tuple[GenerationConfig, dict[str, Any]]],
]


def _default_deserialize_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text[0] not in '[{"tfn-0123456789':
            return value
        try:
            return json.loads(text)
        except Exception:
            return value

    if isinstance(value, dict):
        return dict(value)

    if isinstance(value, list):
        return list(value)

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value

    return value


def load_hf_records(
    dataset_name: str,
    default_config: GenerationConfig,
    row_to_eval_items: RowToEvalItems,
    dataset_config_name: str | None = None,
    split: str = "validation",
    seed: int | None = None,
    deserialize_columns: list[str] | None = None,
    column_deserializers: dict[str, Callable[[Any], Any]] | None = None,
) -> list[tuple[GenerationConfig, dict[str, Any]]]:
    """
    Load a HuggingFace split and directly build evaluator items.

    It merges "load rows" + "iterate rows to produce eval items" into one function.
    """
    dataset = load_dataset(dataset_name, name=dataset_config_name, split=split)
    assert isinstance(dataset, Dataset), "Expected a single Dataset split."

    deserialize_set = set(deserialize_columns or [])
    deserializers = column_deserializers or {}

    result: list[tuple[GenerationConfig, dict[str, Any]]] = []
    for row in dataset:
        record = dict(row)
        for column in deserialize_set:
            if column not in record:
                continue
            if column in deserializers:
                record[column] = deserializers[column](record[column])
            else:
                record[column] = _default_deserialize_value(record[column])

        default_config = default_config.model_copy(deep=True)
        if default_config.seed is None and seed is not None:
            default_config.seed = seed
        result.extend(row_to_eval_items(record, default_config))

    return result
