from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ...core.schema import GenerationConfig
from ..loader import RowToEvalItems, load_hf_records


def load_ella_hf_records(
    dataset_name: str,
    default_config: GenerationConfig,
    row_to_eval_items: RowToEvalItems,
    split: str,
) -> list[tuple[GenerationConfig, dict[str, Any]]]:
    """Load ELLA through datasets, with a raw Hub snapshot fallback.

    `hf download` stores the parquet snapshot but does not populate the Arrow cache
    expected by `datasets` offline mode. Falling back to that parquet keeps the
    evaluator runnable without network access after the benchmark was downloaded.
    """

    try:
        return load_hf_records(
            dataset_name=dataset_name,
            default_config=default_config,
            row_to_eval_items=row_to_eval_items,
            split=split,
            deserialize_columns=["questions"],
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
            parquet_path = snapshot_path / "data" / f"{split}.parquet"
            if not parquet_path.is_file():
                matches = list(snapshot_path.glob(f"**/{split}.parquet"))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Could not identify {split}.parquet in {snapshot_path}."
                    )
                parquet_path = matches[0]

            import pandas as pd

            frame = pd.read_parquet(parquet_path)
            result: list[tuple[GenerationConfig, dict[str, Any]]] = []
            for row in frame.to_dict(orient="records"):
                config = default_config.model_copy(deep=True)
                result.extend(row_to_eval_items(dict(row), config))
            return result
        except Exception as fallback_error:
            raise hub_error from fallback_error
