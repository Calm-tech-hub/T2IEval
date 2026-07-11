import math
from functools import partial
from typing import Any

from pydantic import Field

from ...core.benchmark import BenchmarkSample, SampleEvaluation
from ...core.registry import register_evaluator
from ...core.schema import GenerationConfig
from .. import aggregator, processor
from ..aggregator import Reduction
from ..simple_evaluator import SimpleEvalConfig, SimpleEvaluator
from .loader import load_genaibench_hf_records
from .scorer import ClipFlanT5Scorer, GenAIBenchScorer

SUPPORTED_CONFIGS = {"image_1600", "image_527"}


class GenAIBenchGenerationConfig(GenerationConfig):
    steps: int = 50
    guidance_scale: float = 9.0
    seed: int | None = 42
    width: int | None = 512
    height: int | None = 512
    num_images_per_prompt: int = 1


class GenAIBenchConfig(SimpleEvalConfig):
    dataset_name: str = "Vertsineu/geneaibench"
    dataset_config_name: str = "image_1600"
    split: str = "test_v1"
    num_samples: int | None = None
    score_batch_size: int = 16

    scorer_model: str = "clip-flant5-xxl"
    scorer_version: str = "1.1"
    scorer_cache_dir: str | None = None
    question_template: str | None = None
    answer_template: str | None = None

    generation_config: GenAIBenchGenerationConfig = Field(
        default_factory=GenAIBenchGenerationConfig
    )


def _native_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _genaibench_row_to_eval_items(
    row: dict[str, Any],
    config: GenerationConfig,
) -> list[BenchmarkSample]:
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"GenAI-Bench row has no prompt: {row!r}")

    generation_config = config.model_copy(deep=True)
    generation_config.prompt = prompt
    skills = [str(skill).strip() for skill in _native_list(row.get("skills"))]
    skills = [skill for skill in skills if skill]
    metadata = {
        "prompt_id": str(row.get("prompt_id", row.get("prompt_idx", ""))),
        "prompt_idx": int(row.get("prompt_idx", row.get("idx", 0))),
        "prompt": prompt,
        "prompt_zh": str(row.get("prompt_zh", "")),
        "skills": skills,
    }
    sample_id = metadata["prompt_id"] or str(metadata["prompt_idx"])
    return [
        BenchmarkSample(
            sample_id=sample_id,
            prompt=prompt,
            generation_config=generation_config,
            metadata=metadata,
        )
    ]


class GenAIBenchPostprocessor:
    """Compute VQAScore for every generated image/prompt pair."""

    def __init__(
        self,
        device: str,
        scorer_model: str,
        scorer_version: str,
        score_batch_size: int,
        scorer_cache_dir: str | None = None,
        question_template: str | None = None,
        answer_template: str | None = None,
        scorer: GenAIBenchScorer | None = None,
    ):
        self.device = str(device)
        self.scorer_model = scorer_model
        self.scorer_version = scorer_version
        self.score_batch_size = score_batch_size
        self.scorer_cache_dir = scorer_cache_dir
        self.question_template = question_template
        self.answer_template = answer_template
        self.scorer = scorer
        self._owns_scorer = scorer is None

    def _load_scorer(self) -> GenAIBenchScorer:
        if self.scorer is None:
            print(
                f"Loading GenAI-Bench scorer ({self.scorer_model}, "
                f"t2v-metrics={self.scorer_version})..."
            )
            self.scorer = ClipFlanT5Scorer(
                model_name=self.scorer_model,
                device=self.device,
                cache_dir=self.scorer_cache_dir,
                required_version=self.scorer_version,
                question_template=self.question_template,
                answer_template=self.answer_template,
            )
            print("GenAI-Bench scorer loaded.")
        return self.scorer

    def _unload_scorer(self) -> None:
        if self.scorer is not None and self._owns_scorer:
            close = getattr(self.scorer, "close", None)
            if callable(close):
                close()
            self.scorer = None

    def __call__(
        self, eval_results: list[SampleEvaluation]
    ) -> list[SampleEvaluation]:
        if not eval_results:
            return []

        flat_images = []
        flat_prompts = []
        owners: list[int] = []
        for result_index, item in enumerate(eval_results):
            generation_result = item.generation
            metadata = item.metadata
            for image in generation_result.images:
                flat_images.append(image)
                flat_prompts.append(metadata["prompt"])
                owners.append(result_index)

        scorer = self._load_scorer()
        try:
            flat_scores = scorer.score_pairs(
                flat_images,
                flat_prompts,
                batch_size=self.score_batch_size,
            )
        finally:
            self._unload_scorer()

        if len(flat_scores) != len(flat_images):
            raise RuntimeError(
                "GenAI-Bench scorer returned a different number of scores than images."
            )

        scores_by_result: list[list[float]] = [[] for _ in eval_results]
        for owner, score in zip(owners, flat_scores, strict=True):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(
                    f"GenAI-Bench scorer returned non-finite score {value}."
                )
            scores_by_result[owner].append(value)

        processed = []
        for result_index, item in enumerate(eval_results):
            metadata = item.metadata
            per_image_scores = scores_by_result[result_index]
            prompt_score = (
                float(sum(per_image_scores) / len(per_image_scores))
                if per_image_scores
                else 0.0
            )
            metadata["per_image_scores"] = per_image_scores
            metadata["num_images"] = len(per_image_scores)
            metadata["score"] = prompt_score
            metadata["scorer_model"] = self.scorer_model

            skill_metrics = [
                {"skill": skill, "score": prompt_score} for skill in metadata["skills"]
            ]
            if metadata["skills"]:
                skill_metrics.append({"skill": "all", "score": prompt_score})
            metadata["skill_metrics"] = skill_metrics
            metadata["overall_metrics"] = (
                [{"score": prompt_score}] if metadata["skills"] else []
            )
            processed.append(item)

        return processed


@register_evaluator("genaibench")
class GenAIBenchEvaluator(SimpleEvaluator):
    def __init__(self, **kwargs):
        config = GenAIBenchConfig(**kwargs)
        if config.dataset_config_name not in SUPPORTED_CONFIGS:
            raise ValueError(
                f"Unsupported GenAI-Bench config {config.dataset_config_name!r}; "
                f"expected one of {sorted(SUPPORTED_CONFIGS)}."
            )
        if config.score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive.")

        config.loader = partial(
            load_genaibench_hf_records,
            dataset_name=config.dataset_name,
            dataset_config_name=config.dataset_config_name,
            default_config=config.generation_config,
            row_to_eval_items=_genaibench_row_to_eval_items,
            split=config.split,
        )
        config.preprocessor = []
        if config.num_samples is not None:
            config.preprocessor.append(
                partial(
                    processor.sample,
                    seed=config.seed,
                    sample_size=config.num_samples,
                )
            )

        device = (
            config.accelerator.device
            if config.accelerator is not None
            else config.device
        )
        config.postprocessor = [
            GenAIBenchPostprocessor(
                device=str(device),
                scorer_model=config.scorer_model,
                scorer_version=config.scorer_version,
                score_batch_size=config.score_batch_size,
                scorer_cache_dir=config.scorer_cache_dir,
                question_template=config.question_template,
                answer_template=config.answer_template,
            )
        ]
        config.aggregator = [
            partial(
                aggregator.aggregate_metric_from_list,
                list_key="skill_metrics",
                value_key="score",
                output_key="task_scores",
                reduction=Reduction.MEAN,
                group_by="skill",
            ),
            partial(
                aggregator.aggregate_metric_from_list,
                list_key="overall_metrics",
                value_key="score",
                output_key="score",
                reduction=Reduction.MEAN,
            ),
        ]

        super().__init__(config)
