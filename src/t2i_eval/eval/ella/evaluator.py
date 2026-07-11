import csv
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import Field

from ...core.registry import register_evaluator
from ...core.schema import GenerationConfig, GenerationResult
from .. import processor
from ..simple_evaluator import SimpleEvalConfig, SimpleEvaluator
from .loader import load_ella_hf_records
from .scorer import MPlugVQAScorer, VQAScorer, apply_question_dependencies

PAPER_CATEGORIES = ["global", "entity", "attribute", "relation", "other"]


class EllaGenerationConfig(GenerationConfig):
    """Default SD1.5 generation settings documented for the ELLA task."""

    steps: int = 50
    guidance_scale: float = 12.0
    width: int | None = 512
    height: int | None = 512
    num_images_per_prompt: int = 4


class EllaConfig(SimpleEvalConfig):
    """Configuration for the framework-native ELLA/DPG-Bench evaluator."""

    dataset_name: str = "Vertsineu/ella"
    split: str = "validation"
    csv_path: str | None = None
    num_samples: int | None = None
    seed: int = 1001

    vqa_model_id: str = "damo/mplug_visual-question-answering_coco_large_en"
    missing_dependency_policy: str = "zero"
    category_image_policy: str = "official_last_image"

    generation_config: EllaGenerationConfig = Field(
        default_factory=EllaGenerationConfig
    )


def _as_native_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
                return _as_native_list(decoded)
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _dependency_ids(value: Any) -> list[int]:
    result: list[int] = []
    for item in _as_native_list(value):
        if item is None or (isinstance(item, float) and math.isnan(item)):
            continue
        result.append(int(item))
    return result or [0]


def _category(value: Any, default: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip().lower()
    return text if text and text != "nan" else default


def _normalize_question(question: dict[str, Any]) -> dict[str, Any]:
    qid_value = question.get("qid", question.get("proposition_id"))
    text_value = question.get("question", question.get("question_natural_language", ""))
    if qid_value is None or not str(text_value).strip():
        raise ValueError(f"Invalid ELLA question record: {question!r}")

    broad = _category(question.get("category_broad"), "other")
    detailed = _category(question.get("category_detailed"), "unspecified")
    return {
        "qid": int(qid_value),
        "question": str(text_value).strip(),
        "dependency": _dependency_ids(
            question.get("dependency", question.get("dependencies"))
        ),
        "category_broad": broad,
        "category_detailed": detailed,
        "tuple": str(question.get("tuple", "")).strip(),
    }


def _ella_row_to_eval_items(
    row: dict[str, Any],
    config: GenerationConfig,
) -> list[tuple[GenerationConfig, dict[str, Any]]]:
    prompt = str(row.get("prompt", row.get("text", ""))).strip()
    if not prompt:
        raise ValueError(f"ELLA row has no prompt: {row!r}")

    raw_questions = _as_native_list(row.get("questions"))
    questions = [_normalize_question(dict(question)) for question in raw_questions]
    if not questions:
        raise ValueError(f"ELLA row has no questions: {row!r}")

    generation_config = config.model_copy(deep=True)
    generation_config.prompt = prompt
    metadata = {
        "item_id": str(row.get("item_id", row.get("id", ""))),
        "prompt": prompt,
        "questions": questions,
    }
    return [(generation_config, metadata)]


def _load_csv_records(
    csv_path: str,
    default_config: GenerationConfig,
) -> list[tuple[GenerationConfig, dict[str, Any]]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"ELLA CSV not found: {path}")

    grouped: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            record = grouped.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "prompt": row.get("text", ""),
                    "questions": [],
                },
            )
            record["questions"].append(_normalize_question(row))

    result: list[tuple[GenerationConfig, dict[str, Any]]] = []
    for record in grouped.values():
        result.extend(_ella_row_to_eval_items(record, default_config))
    return result


def load_ella_records(
    dataset_name: str,
    split: str,
    default_config: GenerationConfig,
    csv_path: str | None = None,
) -> list[tuple[GenerationConfig, dict[str, Any]]]:
    if csv_path is not None:
        return _load_csv_records(csv_path, default_config)

    return load_ella_hf_records(
        dataset_name=dataset_name,
        default_config=default_config,
        row_to_eval_items=_ella_row_to_eval_items,
        split=split,
    )


class EllaPostprocessor:
    """Run mPLUG VQA and compute per-image dependency-aware DPG scores."""

    def __init__(
        self,
        device: str,
        vqa_model_id: str,
        missing_dependency_policy: str = "zero",
        scorer: VQAScorer | None = None,
    ):
        self.device = str(device)
        self.vqa_model_id = vqa_model_id
        self.missing_dependency_policy = missing_dependency_policy
        self.scorer = scorer
        self._owns_scorer = scorer is None

    def _load_scorer(self) -> VQAScorer:
        if self.scorer is None:
            print(f"Loading ELLA VQA model ({self.vqa_model_id})...")
            self.scorer = MPlugVQAScorer(
                model_id=self.vqa_model_id,
                device=self.device,
            )
            print("ELLA VQA model loaded.")
        return self.scorer

    def _unload_scorer(self) -> None:
        if self.scorer is not None and self._owns_scorer:
            close = getattr(self.scorer, "close", None)
            if callable(close):
                close()
            self.scorer = None

    def __call__(
        self, eval_results: list[tuple[GenerationResult, dict[str, Any]]]
    ) -> list[tuple[GenerationResult, dict[str, Any]]]:
        scorer = self._load_scorer()
        processed: list[tuple[GenerationResult, dict[str, Any]]] = []

        try:
            for generation_result, metadata in eval_results:
                questions = metadata["questions"]
                image_scores: list[float] = []
                question_metrics: list[dict[str, Any]] = []
                missing_dependencies: list[dict[str, int]] = []

                for image_index, image in enumerate(generation_result.images):
                    answers: dict[int, str] = {}
                    raw_scores: dict[int, float] = {}
                    for question in questions:
                        qid = int(question["qid"])
                        answer = scorer.answer(image, question["question"])
                        answers[qid] = answer
                        raw_scores[qid] = float(answer == "yes")

                    adjusted, validity, missing = apply_question_dependencies(
                        raw_scores,
                        questions,
                        missing_dependency_policy=self.missing_dependency_policy,
                    )
                    missing_dependencies.extend(missing)

                    image_score = (
                        sum(adjusted.values()) / len(adjusted) if adjusted else 0.0
                    )
                    image_scores.append(float(image_score))

                    for question in questions:
                        qid = int(question["qid"])
                        question_metrics.append(
                            {
                                "image_index": image_index,
                                "qid": qid,
                                "answer": answers[qid],
                                "raw_score": raw_scores[qid],
                                "dependency_score": adjusted[qid],
                                "valid": validity[qid],
                                "category_broad": question["category_broad"],
                                "category_detailed": question["category_detailed"],
                            }
                        )

                metadata["per_image_scores"] = image_scores
                metadata["num_images"] = len(image_scores)
                metadata["score"] = (
                    float(sum(image_scores) / len(image_scores))
                    if image_scores
                    else 0.0
                )
                metadata["question_metrics"] = question_metrics
                metadata["missing_dependencies"] = missing_dependencies
                processed.append((generation_result, metadata))
        finally:
            self._unload_scorer()

        return processed


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate_ella_results(
    metadatas: list[dict[str, Any]],
    category_image_policy: str = "official_last_image",
) -> dict[str, Any]:
    if category_image_policy not in {"official_last_image", "all_images"}:
        raise ValueError(
            "category_image_policy must be official_last_image or all_images"
        )

    prompt_scores = [float(metadata.get("score", 0.0)) for metadata in metadatas]
    category_values: dict[str, list[float]] = defaultdict(list)
    detailed_values: dict[str, list[float]] = defaultdict(list)
    question_evaluations = 0

    for metadata in metadatas:
        last_image_index = int(metadata.get("num_images", 0)) - 1
        metrics = metadata.get("question_metrics", [])
        question_evaluations += len(metrics)
        for metric in metrics:
            if (
                category_image_policy == "official_last_image"
                and int(metric["image_index"]) != last_image_index
            ):
                continue

            broad = str(metric["category_broad"])
            detailed = str(metric["category_detailed"])
            raw_score = float(metric["raw_score"])
            category_values[broad].append(raw_score)
            detailed_values[f"{broad} - {detailed}"].append(raw_score)

    ordered_categories = [
        category for category in PAPER_CATEGORIES if category in category_values
    ]
    ordered_categories.extend(sorted(category_values.keys() - set(ordered_categories)))
    task_scores = {
        category: _mean(category_values[category]) for category in ordered_categories
    }
    detailed_scores = {
        category: _mean(values) for category, values in sorted(detailed_values.items())
    }
    score = _mean(prompt_scores)

    return {
        "task_scores": task_scores,
        "task_scores_percent": {
            category: value * 100.0 for category, value in task_scores.items()
        },
        "category_scores_l2": detailed_scores,
        "category_scores_l2_percent": {
            category: value * 100.0 for category, value in detailed_scores.items()
        },
        "score": score,
        "score_percent": score * 100.0,
        "num_prompts": len(metadatas),
        "num_images": sum(int(metadata.get("num_images", 0)) for metadata in metadatas),
        "num_question_evaluations": question_evaluations,
        "missing_dependency_references": sum(
            len(metadata.get("missing_dependencies", [])) for metadata in metadatas
        ),
        "category_image_policy": category_image_policy,
    }


@register_evaluator("ella")
class EllaEvaluator(SimpleEvaluator):
    """Framework-native evaluator for ELLA's DPG-Bench."""

    def __init__(self, **kwargs):
        config = EllaConfig(**kwargs)
        if config.missing_dependency_policy not in {"zero", "ignore", "error"}:
            raise ValueError(
                "missing_dependency_policy must be one of: zero, ignore, error"
            )
        if config.category_image_policy not in {
            "official_last_image",
            "all_images",
        }:
            raise ValueError(
                "category_image_policy must be official_last_image or all_images"
            )

        config.loader = partial(
            load_ella_records,
            dataset_name=config.dataset_name,
            split=config.split,
            default_config=config.generation_config,
            csv_path=config.csv_path,
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
            EllaPostprocessor(
                device=str(device),
                vqa_model_id=config.vqa_model_id,
                missing_dependency_policy=config.missing_dependency_policy,
            )
        ]
        config.aggregator = [
            partial(
                aggregate_ella_results,
                category_image_policy=config.category_image_policy,
            )
        ]

        super().__init__(config)
