from collections.abc import Callable
from functools import partial
from typing import Any

from t2i_eval.core.evaluator import BaseEvaluator
from t2i_eval.core.model import BaseModel
from t2i_eval.core.schema import EvaluatorConfig, GenerationConfig, GenerationResult


def _default_loader() -> list[tuple[GenerationConfig, dict[str, Any]]]:
    return []


class SimpleEvalConfig(EvaluatorConfig):
    """Configuration for SimpleEvaluator."""

    # loader is a function that loads raw data (e.g., a DataFrame) from a source,
    # e.g., from a file or a dataset, and returns a list of tuples,
    loader: Callable[[], list[tuple[GenerationConfig, dict[str, Any]]]] = (
        _default_loader  # Default to an empty list if no loader is provided
    )

    # preprocessor is a function that takes the raw data loaded by loader (e.g., a DataFrame)
    # and returns a list of tuples,
    # where each tuple contains a GenerationConfig and a dict of additional metadata
    preprocessor: list[
        Callable[
            [list[tuple[GenerationConfig, dict[str, Any]]]],
            list[tuple[GenerationConfig, dict[str, Any]]],
        ]
    ] = []

    # postprocessor is a series of functions that takes partial evaluation results (list of tuples of GenerationResult and metadata)
    # and returns a list of tuples of GenerationResult and metadata containing the final evaluation results for each sample.
    # this function indivually evaluates each generated image
    postprocessor: list[
        Callable[
            [list[tuple[GenerationResult, dict[str, Any]]]],
            list[tuple[GenerationResult, dict[str, Any]]],
        ]
    ] = []

    # aggregator is a series of functions that takes metadata list as input
    # and each independently returns a Dict[str, Any].
    # SimpleEvaluator merges these dict outputs as final evaluation result.
    aggregator: list[Callable[[list[dict[str, Any]]], dict[str, Any]]] = []


class SimpleEvaluator(BaseEvaluator):
    """A simple evaluator that does nothing."""

    def __init__(self, config: SimpleEvalConfig):
        self.config = config
        if self.config.sample_dir is not None:
            from .postprocessor import save_results

            self.config.postprocessor.append(
                partial(save_results, sample_dir=self.config.sample_dir),
            )  # Save samples before other postprocessors.

    def evaluate(self, model: BaseModel) -> dict[str, Any] | None:
        # 1. Load raw data using the provided loader function
        raw_data = self.config.loader()

        # 2. Preprocess the raw data into GenerationConfigs and metadata
        processed_data = raw_data
        for preprocessor in self.config.preprocessor:
            processed_data = preprocessor(processed_data)

        if self.config.accelerator is not None:
            model.enable_accelerator(
                self.config.accelerator
            )  # Enable accelerator if provided

            # 3. Split data across ranks.
            process_index = self.config.accelerator.process_index
            num_processes = self.config.accelerator.num_processes
            local_data = processed_data[process_index::num_processes]

            # 4. Generate images for local GenerationConfigs
            local_configs = [item[0] for item in local_data]
            local_gen_results = model.generate_batch(local_configs)
            model.unload()  # Unload model after generation to free up resources for evaluation

            # 5. Postprocess the generation results with metadata to get evaluation results for each sample
            # (Each Rank)
            local_eval_results = list(
                zip(local_gen_results, [item[1] for item in local_data])
            )  # Combine GenerationResults with metadata
            for postprocessor in self.config.postprocessor:
                local_eval_results = postprocessor(local_eval_results)

            # 6. Aggregate the evaluation results across samples to compute overall metrics
            # (Rank 0 Only)
            from accelerate.utils import gather_object

            # Gather metadata-only eval results to rank 0.
            local_metadatas = [metadata for _, metadata in local_eval_results]
            all_metadatas = gather_object(local_metadatas)

            # Only rank 0 performs aggregation
            if self.config.accelerator.is_main_process:
                merged_result: dict[str, Any] = {}
                for aggregator in self.config.aggregator:
                    merged_result.update(aggregator(all_metadatas))
                return merged_result
            else:
                return None
        else:
            # 3. Generate images for all GenerationConfigs sequentially
            configs = [item[0] for item in processed_data]
            gen_results = model.generate_batch(configs)
            model.unload()  # Unload model after generation to free up resources for evaluation

            # 4. Postprocess the generation results with metadata to get evaluation results for each sample
            eval_results = list(
                zip(gen_results, [item[1] for item in processed_data])
            )  # Combine GenerationResults with metadata
            for postprocessor in self.config.postprocessor:
                eval_results = postprocessor(eval_results)
            metadatas = [metadata for _, metadata in eval_results]

            # 5. Aggregate the evaluation results across samples to compute overall metrics
            # No accelerator, process all results locally
            merged_result: dict[str, Any] = {}
            for aggregator in self.config.aggregator:
                merged_result.update(aggregator(metadatas))
            return merged_result
