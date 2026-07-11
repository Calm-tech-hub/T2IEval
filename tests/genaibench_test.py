from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from t2i_eval.core.benchmark import SampleEvaluation
from t2i_eval.core.registry import get_evaluator_class
from t2i_eval.core.schema import GenerationConfig, GenerationResult
from t2i_eval.eval.genaibench.evaluator import (
    GenAIBenchConfig,
    GenAIBenchEvaluator,
    GenAIBenchPostprocessor,
    _genaibench_row_to_eval_items,
)
from t2i_eval.eval.genaibench.scorer import ClipFlanT5Scorer, _load_clip_t5_module


class FakeScorer:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def score_pairs(
        self,
        images: list[Image.Image],
        texts: list[str],
        batch_size: int,
    ) -> list[float]:
        assert len(images) == len(texts)
        assert batch_size == 4
        return self.scores


def _metadata(prompt_id: str, prompt: str, skills: list[str]) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "prompt_idx": int(prompt_id),
        "prompt": prompt,
        "prompt_zh": "",
        "skills": skills,
    }


def _evaluation(
    prompt_id: str,
    prompt: str,
    skills: list[str],
    image_count: int = 1,
) -> SampleEvaluation:
    return SampleEvaluation(
        sample_id=prompt_id,
        prompt=prompt,
        generation=GenerationResult(
            images=[Image.new("RGB", (8, 8)) for _ in range(image_count)]
        ),
        metadata=_metadata(prompt_id, prompt, skills),
    )


def test_genaibench_is_registered_and_has_paper_defaults():
    assert get_evaluator_class("genaibench") is not None
    config = GenAIBenchConfig()
    assert config.dataset_config_name == "image_1600"
    assert config.scorer_model == "clip-flant5-xxl"
    assert config.scorer_version == "1.1"
    assert config.generation_config.steps == 50
    assert config.generation_config.num_images_per_prompt == 1


def test_row_converter_preserves_prompt_and_skills():
    items = _genaibench_row_to_eval_items(
        {
            "idx": 7,
            "prompt_id": "00007",
            "prompt_idx": 7,
            "prompt": "A red cube above a blue sphere.",
            "prompt_zh": "",
            "skills": ["attribute", "spatial relation", "basic"],
        },
        GenerationConfig(seed=42),
    )

    assert len(items) == 1
    sample = items[0]
    assert sample.generation_config.prompt == "A red cube above a blue sphere."
    assert sample.metadata["prompt_id"] == "00007"
    assert sample.metadata["skills"] == ["attribute", "spatial relation", "basic"]


def test_postprocessor_scores_images_and_builds_aggregation_lists():
    postprocessor = GenAIBenchPostprocessor(
        device="cpu",
        scorer_model="fake",
        scorer_version="1.1",
        score_batch_size=4,
        scorer=FakeScorer([0.2, 0.4, 0.8, 0.1]),
    )
    results = postprocessor(
        [
            _evaluation("00001", "basic prompt", ["attribute", "basic"], 2),
            _evaluation("00002", "advanced prompt", ["counting", "advanced"]),
            _evaluation("00003", "untagged prompt", []),
        ]
    )

    assert results[0].metadata["score"] == pytest.approx(0.3)
    assert results[1].metadata["score"] == pytest.approx(0.8)
    assert results[2].metadata["score"] == pytest.approx(0.1)
    assert results[0].metadata["skill_metrics"] == [
        {"skill": "attribute", "score": pytest.approx(0.3)},
        {"skill": "basic", "score": pytest.approx(0.3)},
        {"skill": "all", "score": pytest.approx(0.3)},
    ]
    assert results[2].metadata["overall_metrics"] == []


def test_evaluator_reuses_common_aggregators_and_matches_official_scope():
    evaluator = GenAIBenchEvaluator()
    metadatas = [
        {
            "skill_metrics": [
                {"skill": "attribute", "score": 0.3},
                {"skill": "basic", "score": 0.3},
                {"skill": "all", "score": 0.3},
            ],
            "overall_metrics": [{"score": 0.3}],
        },
        {
            "skill_metrics": [
                {"skill": "counting", "score": 0.8},
                {"skill": "advanced", "score": 0.8},
                {"skill": "all", "score": 0.8},
            ],
            "overall_metrics": [{"score": 0.8}],
        },
        {"skill_metrics": [], "overall_metrics": []},
    ]

    evaluations = [
        SampleEvaluation(
            sample_id=str(index),
            generation=GenerationResult(images=[]),
            metadata=metadata,
        )
        for index, metadata in enumerate(metadatas)
    ]
    metrics = {}
    for aggregate in evaluator.config.aggregator:
        metrics.update(aggregate(evaluations))

    assert metrics["task_scores"] == {
        "attribute": 0.3,
        "basic": 0.3,
        "all": pytest.approx(0.55),
        "counting": 0.8,
        "advanced": 0.8,
    }
    assert metrics["score"] == pytest.approx(0.55)


def test_invalid_config_and_batch_size_are_rejected():
    with pytest.raises(ValueError, match="Unsupported GenAI-Bench config"):
        GenAIBenchEvaluator(dataset_config_name="video_800")
    with pytest.raises(ValueError, match="score_batch_size must be positive"):
        GenAIBenchEvaluator(score_batch_size=0)

def test_clip_flant5_scorer_batches_paired_inputs():
    class FakeClipT5Model:
        def __init__(self):
            self.calls = []
            self.offset = 0

        def forward(self, image_paths, texts, **kwargs):
            assert all(Path(path).is_file() for path in image_paths)
            self.calls.append((image_paths, texts, kwargs))
            values = torch.arange(
                self.offset + 1, self.offset + len(image_paths) + 1
            ).float() / 10
            self.offset += len(image_paths)
            return values

    scorer = object.__new__(ClipFlanT5Scorer)
    scorer.score_model = FakeClipT5Model()
    scorer.question_template = "Question: {}"
    scorer.answer_template = "Yes"
    images = [Image.new("RGB", (8, 8)) for _ in range(3)]

    scores = scorer.score_pairs(images, ["one", "two", "three"], batch_size=2)

    assert scores == pytest.approx([0.1, 0.2, 0.3])
    assert [call[1] for call in scorer.score_model.calls] == [
        ["one", "two"],
        ["three"],
    ]
    assert scorer.score_model.calls[0][2] == {
        "question_template": "Question: {}",
        "answer_template": "Yes",
    }


def test_clip_flant5_submodule_imports_without_optional_backends():
    module = _load_clip_t5_module()

    assert module.CLIPT5Model.__name__ == "CLIPT5Model"
    assert "clip-flant5-xxl" in module.CLIP_T5_MODELS
    assert "clip-flant5-xl" in module.CLIP_T5_MODELS
