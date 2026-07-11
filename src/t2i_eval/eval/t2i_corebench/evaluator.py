import copy
import re
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ...core.benchmark import SampleEvaluation
from ...core.registry import register_evaluator
from ...core.schema import GenerationConfig
from ..simple_evaluator import SimpleEvalConfig, SimpleEvaluator
from .loader import (
    ALL_DIMENSIONS,
    COMPOSITION_DIMENSIONS,
    REASONING_DIMENSIONS,
    load_corebench_records,
    normalize_dimensions,
)
from .scorer import (
    CoreBenchJudge,
    JudgeRequest,
    JudgeResponse,
    SubprocessQwenJudge,
)


class T2ICoreBenchGenerationConfig(GenerationConfig):
    """Official Qwen-Image sampling defaults from the benchmark repository."""

    negative_prompt: str | None = " "
    steps: int = 50
    guidance_scale: float = 4.0
    seed: int | None = 0
    width: int | None = 1328
    height: int | None = 1328
    num_images_per_prompt: int = 4
    extra_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"true_cfg_scale": 4.0}
    )


class T2ICoreBenchConfig(SimpleEvalConfig):
    dataset_name: str = "lioooox/T2I-CoReBench"
    data_dir: str | None = None
    dataset_cache_dir: str | None = None
    dimensions: str | list[str] = "all"
    num_prompts: int | None = None

    # Set this to the model-level directory containing C-MI/, C-MA/, ...
    # and run with `model: precomputed` to evaluate existing images.
    image_dir: str | None = None
    strict_images: bool = True

    run_mode: Literal["evaluate", "generate_only"] = "evaluate"
    judge_backend: Literal["qwen_subprocess"] = "qwen_subprocess"
    judge_model: str = "Qwen/Qwen3.5-9B"
    judge_python: str | None = None
    judge_cache_path: str | None = None
    judge_batch_size: int = 64
    judge_max_rounds: int = 3
    judge_initial_max_tokens: int = 512
    judge_gpu_memory_utilization: float = 0.85
    judge_tensor_parallel_size: int = 1
    judge_request_chunk_size: int = 4096
    judge_timeout: float | None = None

    # Injection point used by lightweight tests and alternative local judges.
    judge: Any | None = Field(default=None, exclude=True)

    generation_config: T2ICoreBenchGenerationConfig = Field(
        default_factory=T2ICoreBenchGenerationConfig
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_corebench_results(
    results: list[SampleEvaluation],
) -> dict[str, Any]:
    dimension_values: dict[str, list[float]] = {}
    prompt_ids = set()
    question_count = 0
    valid_question_count = 0
    cached_question_count = 0

    for result in results:
        metadata = result.metadata
        dimension = str(metadata["dimension"])
        prompt_ids.add(str(metadata["prompt_id"]))
        image_score = metadata.get("image_score")
        if image_score is not None:
            dimension_values.setdefault(dimension, []).append(float(image_score))
        for question in metadata.get("checklist", []):
            question_count += 1
            if question.get("score") in (0, 1):
                valid_question_count += 1
            if question.get("cached"):
                cached_question_count += 1

    present_dimensions = {
        str(result.metadata["dimension"]) for result in results
    }
    dimension_scores = {
        dimension: _mean(dimension_values.get(dimension, []))
        for dimension in ALL_DIMENSIONS
        if dimension in present_dimensions
    }
    composition_values = [
        dimension_scores[dimension]
        for dimension in COMPOSITION_DIMENSIONS
        if dimension_scores.get(dimension) is not None
    ]
    reasoning_values = [
        dimension_scores[dimension]
        for dimension in REASONING_DIMENSIONS
        if dimension_scores.get(dimension) is not None
    ]
    available_dimension_values = [
        value for value in dimension_scores.values() if value is not None
    ]
    selected = list(dimension_scores)
    return {
        "score": _mean(available_dimension_values),
        "composition_score": _mean(composition_values),
        "reasoning_score": _mean(reasoning_values),
        "dimension_scores": dimension_scores,
        "evaluated_dimensions": selected,
        "is_partial_evaluation": set(selected) != set(ALL_DIMENSIONS),
        "num_prompts": len(prompt_ids),
        "num_images": len(results),
        "num_questions": question_count,
        "num_valid_answers": valid_question_count,
        "num_invalid_answers": question_count - valid_question_count,
        "valid_answer_rate": (
            valid_question_count / question_count if question_count else 0.0
        ),
        "cached_answer_rate": (
            cached_question_count / question_count if question_count else 0.0
        ),
    }


def aggregate_generation_only(
    results: list[SampleEvaluation],
) -> dict[str, Any]:
    prompt_ids = {str(result.metadata["prompt_id"]) for result in results}
    dimensions = list(
        dict.fromkeys(str(result.metadata["dimension"]) for result in results)
    )
    return {
        "status": "generation_only",
        "evaluated": False,
        "evaluated_dimensions": dimensions,
        "num_prompts": len(prompt_ids),
        "num_images": sum(len(result.generation.images) for result in results),
    }


class T2ICoreBenchPostprocessor:
    def __init__(
        self,
        *,
        judge_model: str,
        judge_python: str | None,
        judge_cache_path: str | None,
        judge_batch_size: int,
        judge_max_rounds: int,
        judge_initial_max_tokens: int,
        judge_gpu_memory_utilization: float,
        judge_tensor_parallel_size: int,
        judge_request_chunk_size: int,
        judge_timeout: float | None,
        judge: CoreBenchJudge | None = None,
    ):
        self.judge_model = judge_model
        self.judge_python = judge_python
        self.judge_cache_path = judge_cache_path
        self.judge_batch_size = judge_batch_size
        self.judge_max_rounds = judge_max_rounds
        self.judge_initial_max_tokens = judge_initial_max_tokens
        self.judge_gpu_memory_utilization = judge_gpu_memory_utilization
        self.judge_tensor_parallel_size = judge_tensor_parallel_size
        self.judge_request_chunk_size = judge_request_chunk_size
        self.judge_timeout = judge_timeout
        self.judge = judge
        self._owns_judge = judge is None
        self.question_records: list[dict[str, Any]] = []

    def _load_judge(self) -> CoreBenchJudge:
        if self.judge is None:
            self.judge = SubprocessQwenJudge(
                model_name=self.judge_model,
                python_executable=self.judge_python,
                cache_path=self.judge_cache_path,
                batch_size=self.judge_batch_size,
                max_rounds=self.judge_max_rounds,
                initial_max_tokens=self.judge_initial_max_tokens,
                gpu_memory_utilization=self.judge_gpu_memory_utilization,
                tensor_parallel_size=self.judge_tensor_parallel_size,
                request_chunk_size=self.judge_request_chunk_size,
                timeout=self.judge_timeout,
            )
        return self.judge

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    def _image_paths(
        self,
        result: SampleEvaluation,
        temporary_dir: Path,
    ) -> list[str]:
        paths = [Path(path) for path in result.image_paths if Path(path).is_file()]
        if not paths:
            paths = [
                Path(path)
                for path in result.generation.debug_info.get("source_paths", [])
                if Path(path).is_file()
            ]
        if paths:
            result.image_paths = [str(path.resolve()) for path in paths]
            return result.image_paths

        result.image_paths = []
        for image_index, image in enumerate(result.generation.images):
            path = temporary_dir / (
                f"{self._safe_name(result.sample_id)}-{image_index}.png"
            )
            image.save(path)
            result.image_paths.append(str(path))
        return result.image_paths

    def __call__(
        self,
        results: list[SampleEvaluation],
    ) -> list[SampleEvaluation]:
        if not results:
            return []

        with tempfile.TemporaryDirectory(prefix="t2i-corebench-images-") as directory:
            temporary_dir = Path(directory)
            requests: list[JudgeRequest] = []
            owners: dict[str, tuple[int, int, int]] = {}
            scored_checklists: list[list[list[dict[str, Any]]]] = []

            for result_index, result in enumerate(results):
                image_paths = self._image_paths(result, temporary_dir)
                base_checklist = result.metadata["checklist"]
                per_image_checklists = []
                for image_index, image_path in enumerate(image_paths):
                    checklist = copy.deepcopy(base_checklist)
                    per_image_checklists.append(checklist)
                    for question_index, question in enumerate(checklist):
                        request_id = (
                            f"{result.sample_id}::image{image_index}::"
                            f"question{question_index}"
                        )
                        owners[request_id] = (
                            result_index,
                            image_index,
                            question_index,
                        )
                        requests.append(
                            JudgeRequest(
                                request_id=request_id,
                                image_path=image_path,
                                prompt=result.prompt,
                                question=str(question["question"]),
                            )
                        )
                scored_checklists.append(per_image_checklists)

            judge = self._load_judge()
            try:
                responses = judge.score(requests)
            finally:
                if self._owns_judge:
                    judge.close()
                    self.judge = None

            response_map: dict[str, JudgeResponse] = {
                response.request_id: response for response in responses
            }
            self.question_records = []
            for request in requests:
                result_index, image_index, question_index = owners[request.request_id]
                response = response_map.get(
                    request.request_id,
                    JudgeResponse(
                        request_id=request.request_id,
                        answer=None,
                        score=None,
                        raw_response="",
                        error="missing_judge_response",
                    ),
                )
                question = scored_checklists[result_index][image_index][question_index]
                question.update(
                    {
                        "answer": response.answer,
                        "score": response.score,
                        "raw_response": response.raw_response,
                        "error": response.error,
                        "cached": response.cached,
                    }
                )
                owner = results[result_index]
                self.question_records.append(
                    {
                        "request_id": request.request_id,
                        "sample_id": owner.sample_id,
                        "prompt_id": owner.metadata["prompt_id"],
                        "dimension": owner.metadata["dimension"],
                        "image_index": owner.metadata["image_index"],
                        "question_index": question_index,
                        "question": question["question"],
                        "tags": question.get("tags", []),
                        "answer": response.answer,
                        "score": response.score,
                        "raw_response": response.raw_response,
                        "error": response.error,
                        "cached": response.cached,
                        "judge_model": self.judge_model,
                    }
                )

            for result_index, result in enumerate(results):
                image_results = []
                for image_index, checklist in enumerate(
                    scored_checklists[result_index]
                ):
                    valid_scores = [
                        int(question["score"])
                        for question in checklist
                        if question.get("score") in (0, 1)
                    ]
                    image_results.append(
                        {
                            "image_index": image_index,
                            "image_path": result.image_paths[image_index],
                            "checklist": checklist,
                            "image_score": _mean(valid_scores),
                            "valid_questions": len(valid_scores),
                            "invalid_questions": len(checklist) - len(valid_scores),
                        }
                    )

                # The loader emits one image per sample. Supporting a list here
                # keeps the postprocessor safe for custom loaders as well.
                all_valid_scores = [
                    int(question["score"])
                    for image_result in image_results
                    for question in image_result["checklist"]
                    if question.get("score") in (0, 1)
                ]
                result.metadata["image_results"] = image_results
                result.metadata["checklist"] = (
                    image_results[0]["checklist"] if image_results else []
                )
                result.metadata["image_score"] = _mean(all_valid_scores)
                result.metadata["judge_model"] = self.judge_model
                result.scores["image_score"] = result.metadata["image_score"]

        return results


@register_evaluator("t2i_corebench")
class T2ICoreBenchEvaluator(SimpleEvaluator):
    def __init__(self, **kwargs):
        config = T2ICoreBenchConfig(**kwargs)
        config.dimensions = normalize_dimensions(config.dimensions)
        if config.run_mode == "generate_only" and config.sample_dir is None:
            raise ValueError(
                "T2I-CoReBench generate_only mode requires output.save_images=true."
            )
        config.loader = partial(
            load_corebench_records,
            data_dir=config.data_dir,
            dataset_name=config.dataset_name,
            cache_dir=config.dataset_cache_dir,
            dimensions=config.dimensions,
            default_config=config.generation_config,
            num_prompts=config.num_prompts,
            image_dir=config.image_dir,
            strict_images=config.strict_images,
        )

        self.corebench_postprocessor: T2ICoreBenchPostprocessor | None = None
        if config.run_mode == "generate_only":
            config.postprocessor = []
            config.aggregator = [aggregate_generation_only]
        else:
            cache_path = config.judge_cache_path
            if cache_path is None and config.sample_dir is not None:
                cache_path = str(Path(config.sample_dir).parent / "judge_cache.jsonl")
            self.corebench_postprocessor = T2ICoreBenchPostprocessor(
                judge_model=config.judge_model,
                judge_python=config.judge_python,
                judge_cache_path=cache_path,
                judge_batch_size=config.judge_batch_size,
                judge_max_rounds=config.judge_max_rounds,
                judge_initial_max_tokens=config.judge_initial_max_tokens,
                judge_gpu_memory_utilization=config.judge_gpu_memory_utilization,
                judge_tensor_parallel_size=config.judge_tensor_parallel_size,
                judge_request_chunk_size=config.judge_request_chunk_size,
                judge_timeout=config.judge_timeout,
                judge=config.judge,
            )
            config.postprocessor = [self.corebench_postprocessor]
            config.aggregator = [aggregate_corebench_results]
        super().__init__(config)
        self.artifact_records: dict[str, list[dict[str, Any]]] = {}

    def evaluate(self, model) -> dict[str, Any] | None:
        if self.config.image_dir is not None and not getattr(
            model, "supports_precomputed_images", False
        ):
            raise ValueError(
                "T2I-CoReBench `image_dir` requires `model: precomputed`; "
                "otherwise existing images could accidentally be regenerated."
            )
        if self.config.image_dir is None and getattr(
            model, "supports_precomputed_images", False
        ):
            raise ValueError(
                "The precomputed model requires T2I-CoReBench `image_dir`."
            )

        metrics = super().evaluate(model)
        if self.corebench_postprocessor is not None:
            self.artifact_records["questions.jsonl"] = (
                self.corebench_postprocessor.question_records
            )
        return metrics


__all__ = [
    "T2ICoreBenchConfig",
    "T2ICoreBenchEvaluator",
    "T2ICoreBenchGenerationConfig",
    "T2ICoreBenchPostprocessor",
    "aggregate_corebench_results",
]
