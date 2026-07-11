from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import Field
from tqdm.auto import tqdm

from t2i_eval.core.benchmark import BenchmarkSample, SampleEvaluation
from t2i_eval.core.evaluator import BaseEvaluator
from t2i_eval.core.model import BaseModel
from t2i_eval.core.schema import EvaluatorConfig, GenerationConfig, GenerationResult


def _default_loader() -> list[BenchmarkSample]:
    return []


class SimpleEvalConfig(EvaluatorConfig):
    """Configuration for SimpleEvaluator."""

    # loader is a function that loads raw data (e.g., a DataFrame) from a source,
    # e.g., from a file or a dataset, and returns a list of tuples,
    loader: Callable[[], list[BenchmarkSample]] = (
        _default_loader  # Default to an empty list if no loader is provided
    )

    # preprocessor is a function that takes the raw data loaded by loader (e.g., a DataFrame)
    # and returns a list of tuples,
    # where each tuple contains a GenerationConfig and a dict of additional metadata
    preprocessor: list[
        Callable[
            [list[BenchmarkSample]],
            list[BenchmarkSample],
        ]
    ] = Field(default_factory=list)

    # postprocessor is a series of functions that takes partial evaluation results (list of tuples of GenerationResult and metadata)
    # and returns a list of tuples of GenerationResult and metadata containing the final evaluation results for each sample.
    # this function indivually evaluates each generated image
    postprocessor: list[
        Callable[
            [list[SampleEvaluation]],
            list[SampleEvaluation],
        ]
    ] = Field(default_factory=list)

    # aggregator is a series of functions that takes metadata list as input
    # and each independently returns a Dict[str, Any].
    # SimpleEvaluator merges these dict outputs as final evaluation result.
    aggregator: list[Callable[[list[SampleEvaluation]], dict[str, Any]]] = Field(
        default_factory=list
    )

    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)


class SimpleEvaluator(BaseEvaluator):
    """A simple evaluator that does nothing."""

    def __init__(self, config: SimpleEvalConfig):
        self.config = config
        self.sample_records: list[dict[str, Any]] = []

    @staticmethod
    def _attach_indices(samples: list[BenchmarkSample]) -> None:
        for index, sample in enumerate(samples):
            sample.metadata["_t2i_eval_sample_index"] = index

    def _load_saved(self, sample: BenchmarkSample) -> SampleEvaluation | None:
        if not self.config.resume or self.config.sample_dir is None:
            return None
        index = int(sample.metadata["_t2i_eval_sample_index"])
        sample_dir = Path(self.config.sample_dir) / f"{index:06d}"
        if not (sample_dir / "complete.marker").is_file():
            return None
        image_paths = sorted(sample_dir.glob("*.png"))
        expected = sample.generation_config.num_images_per_prompt
        if len(image_paths) != expected:
            return None
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.copy())
        return SampleEvaluation(
            sample_id=sample.sample_id,
            prompt=sample.prompt,
            generation=GenerationResult(
                images=images,
                debug_info={"resumed": True},
            ),
            image_paths=[str(path) for path in image_paths],
            metadata=sample.metadata,
        )

    def _generate(
        self,
        model: BaseModel,
        samples: list[BenchmarkSample],
    ) -> list[SampleEvaluation]:
        if self.config.sample_dir is None:
            generation_results = model.generate_batch(
                [sample.generation_config for sample in samples]
            )
            return [
                SampleEvaluation(
                    sample_id=sample.sample_id,
                    prompt=sample.prompt,
                    generation=generation,
                    metadata=sample.metadata,
                )
                for sample, generation in zip(samples, generation_results, strict=True)
            ]

        # Persistence is intentionally sample-granular: every completed image
        # set is durable before the next expensive generation begins.
        from .postprocessor import save_results

        evaluations: list[SampleEvaluation] = []
        for sample in tqdm(samples, desc="Generating", unit="prompt"):
            evaluation = self._load_saved(sample)
            if evaluation is None:
                generation = model.generate(sample.generation_config)
                evaluation = SampleEvaluation(
                    sample_id=sample.sample_id,
                    prompt=sample.prompt,
                    generation=generation,
                    metadata=sample.metadata,
                )
                save_results(
                    [evaluation],
                    sample_dir=self.config.sample_dir,
                    use_shared_index=False,
                )
            evaluations.append(evaluation)
        return evaluations

    @staticmethod
    def _records(results: list[SampleEvaluation]) -> list[dict[str, Any]]:
        records = []
        for result in results:
            metadata = {
                key: value
                for key, value in result.metadata.items()
                if not key.startswith("_t2i_eval_")
            }
            scores = {
                key: value
                for key, value in metadata.items()
                if "score" in key or key in {"correct", "prompt_correct"}
            }
            records.append(
                {
                    "sample_id": result.sample_id,
                    "prompt": result.prompt,
                    "image_paths": result.image_paths,
                    "scores": scores,
                    "metadata": metadata,
                    "error": result.error,
                    "debug_info": result.generation.debug_info,
                }
            )
        return records

    def evaluate(self, model: BaseModel) -> dict[str, Any] | None:
        # 1. Load raw data using the provided loader function
        raw_data = self.config.loader()
        if any(not isinstance(item, BenchmarkSample) for item in raw_data):
            raise TypeError(
                "Benchmark loaders must return BenchmarkSample objects."
            )

        # 2. Preprocess the raw data into GenerationConfigs and metadata
        processed_data = raw_data
        for preprocessor in self.config.preprocessor:
            processed_data = preprocessor(processed_data)
        self._attach_indices(processed_data)

        if self.config.accelerator is not None:
            if not getattr(model, "_loaded", False):
                model.enable_accelerator(self.config.accelerator)

            # 3. Split data across ranks.
            process_index = self.config.accelerator.process_index
            num_processes = self.config.accelerator.num_processes
            local_data = processed_data[process_index::num_processes]

            # 4. Generate images for local GenerationConfigs
            local_eval_results = self._generate(model, local_data)
            model.unload()  # Unload model after generation to free up resources for evaluation

            # 5. Postprocess the generation results with metadata to get evaluation results for each sample
            # (Each Rank)
            for postprocessor in self.config.postprocessor:
                local_eval_results = postprocessor(local_eval_results)

            # 6. Aggregate the evaluation results across samples to compute overall metrics
            # (Rank 0 Only)
            from accelerate.utils import gather_object

            # Gather metadata-only eval results to rank 0.
            all_results = gather_object(local_eval_results)

            # Only rank 0 performs aggregation
            if self.config.accelerator.is_main_process:
                merged_result: dict[str, Any] = {}
                for aggregator in self.config.aggregator:
                    merged_result.update(aggregator(all_results))
                self.sample_records = self._records(all_results)
                return merged_result
            else:
                return None
        else:
            # 3. Generate images for all GenerationConfigs sequentially
            eval_results = self._generate(model, processed_data)
            model.unload()  # Unload model after generation to free up resources for evaluation

            # 4. Postprocess the generation results with metadata to get evaluation results for each sample
            for postprocessor in self.config.postprocessor:
                eval_results = postprocessor(eval_results)

            # 5. Aggregate the evaluation results across samples to compute overall metrics
            # No accelerator, process all results locally
            merged_result: dict[str, Any] = {}
            for aggregator in self.config.aggregator:
                merged_result.update(aggregator(eval_results))
            self.sample_records = self._records(eval_results)
            return merged_result
