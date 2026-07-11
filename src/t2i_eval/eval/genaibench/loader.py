from pathlib import Path

from huggingface_hub import snapshot_download

from ...core.benchmark import BenchmarkSample
from ...core.schema import GenerationConfig
from ..loader import RowToEvalItems, load_hf_records


def load_genaibench_hf_records(
    dataset_name: str,
    dataset_config_name: str,
    default_config: GenerationConfig,
    row_to_eval_items: RowToEvalItems,
    split: str,
) -> list[BenchmarkSample]:
    """Load GenAI-Bench, including raw parquet downloaded with `hf download`."""

    try:
        return load_hf_records(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            default_config=default_config,
            row_to_eval_items=row_to_eval_items,
            split=split,
            deserialize_columns=["skills"],
        )
    except ConnectionError as hub_error:
        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=dataset_name,
                    repo_type="dataset",
                    local_files_only=True,
                )
            )
            config_path = snapshot_path / dataset_config_name
            parquet_paths = sorted(config_path.glob(f"{split}*.parquet"))
            if not parquet_paths:
                parquet_paths = sorted(
                    snapshot_path.glob(f"**/{dataset_config_name}/**/{split}*.parquet")
                )
            if not parquet_paths:
                raise FileNotFoundError(
                    f"Could not find {dataset_config_name}/{split} parquet files "
                    f"in {snapshot_path}."
                )

            import pandas as pd

            frame = pd.concat(
                [pd.read_parquet(path) for path in parquet_paths],
                ignore_index=True,
            )
            result: list[BenchmarkSample] = []
            for row in frame.to_dict(orient="records"):
                config = default_config.model_copy(deep=True)
                result.extend(row_to_eval_items(dict(row), config))
            return result
        except Exception as fallback_error:
            raise hub_error from fallback_error
