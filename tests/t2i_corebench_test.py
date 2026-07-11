import json
from pathlib import Path

import pytest
from PIL import Image

from t2i_eval.core.benchmark import SampleEvaluation
from t2i_eval.core.registry import get_evaluator_class, get_model_class
from t2i_eval.core.schema import GenerationConfig, GenerationResult
from t2i_eval.eval.t2i_corebench.evaluator import (
    T2ICoreBenchEvaluator,
    aggregate_corebench_results,
)
from t2i_eval.eval.t2i_corebench.loader import (
    ALL_DIMENSIONS,
    load_corebench_records,
    normalize_dimensions,
)
from t2i_eval.eval.t2i_corebench.scorer import JudgeResponse, parse_binary_answer
from t2i_eval.model.precomputed_model import PrecomputedImageModel


def _write_dimension(root: Path, dimension: str, questions: list[str]) -> None:
    payload = {
        f"{dimension}-001": {
            "Main Class": "Composition" if dimension.startswith("C-") else "Reasoning",
            "Sub Class": "Fixture",
            "Prompt": f"A fixture prompt for {dimension}.",
            "Checklist": [
                {"question": question, "tags": ["fixture"]}
                for question in questions
            ],
            "Remark": "test",
        }
    }
    (root / f"{dimension}.json").write_text(json.dumps(payload))


class FakeJudge:
    model_name = "fake-qwen"

    def score(self, requests):
        responses = []
        for index, request in enumerate(requests):
            score = 1 if index % 2 == 0 else 0
            responses.append(
                JudgeResponse(
                    request_id=request.request_id,
                    answer="yes" if score else "no",
                    score=score,
                    raw_response="yes" if score else "no",
                )
            )
        return responses

    def close(self):
        return None


def test_corebench_registration_and_all_dimensions():
    assert get_evaluator_class("t2i_corebench") is not None
    assert get_model_class("precomputed") is not None
    assert normalize_dimensions("all") == list(ALL_DIMENSIONS)
    assert normalize_dimensions("C-MI, R-CR") == ["C-MI", "R-CR"]
    with pytest.raises(ValueError, match="Unknown T2I-CoReBench"):
        normalize_dimensions(["unknown"])


def test_loader_expands_prompts_into_reproducible_image_samples(tmp_path):
    _write_dimension(tmp_path, "C-MI", ["First?", "Second?"])
    samples = load_corebench_records(
        data_dir=str(tmp_path),
        dataset_name="unused",
        cache_dir=None,
        dimensions=["C-MI"],
        default_config=GenerationConfig(seed=10, num_images_per_prompt=2),
        num_prompts=None,
        image_dir=None,
        strict_images=True,
    )

    assert [sample.sample_id for sample in samples] == ["C-MI-001-0", "C-MI-001-1"]
    assert [sample.generation_config.seed for sample in samples] == [10, 11]
    assert all(sample.generation_config.num_images_per_prompt == 1 for sample in samples)
    assert samples[0].metadata["checklist"][1]["question"] == "Second?"


def test_precomputed_loader_and_model_read_official_style_paths(tmp_path):
    data_dir = tmp_path / "data"
    image_dir = tmp_path / "images"
    data_dir.mkdir()
    (image_dir / "C-MI").mkdir(parents=True)
    _write_dimension(data_dir, "C-MI", ["Visible?"])
    image_path = image_dir / "C-MI" / "C-MI-001-0.png"
    Image.new("RGB", (8, 8), "red").save(image_path)

    samples = load_corebench_records(
        data_dir=str(data_dir),
        dataset_name="unused",
        cache_dir=None,
        dimensions=["C-MI"],
        default_config=GenerationConfig(num_images_per_prompt=1),
        num_prompts=None,
        image_dir=str(image_dir),
        strict_images=True,
    )
    model = PrecomputedImageModel(device="cpu")
    result = model.generate(samples[0].generation_config)

    assert len(result.images) == 1
    assert result.images[0].getpixel((0, 0)) == (255, 0, 0)
    assert result.debug_info["source_paths"] == [str(image_path.resolve())]


def test_evaluator_scores_existing_images_and_exposes_question_records(tmp_path):
    data_dir = tmp_path / "data"
    image_dir = tmp_path / "images"
    data_dir.mkdir()
    (image_dir / "C-MI").mkdir(parents=True)
    _write_dimension(data_dir, "C-MI", ["One?", "Two?"])
    Image.new("RGB", (8, 8), "white").save(
        image_dir / "C-MI" / "C-MI-001-0.png"
    )

    evaluator = T2ICoreBenchEvaluator(
        data_dir=str(data_dir),
        dimensions=["C-MI"],
        image_dir=str(image_dir),
        judge_model="fake-qwen",
        judge=FakeJudge(),
        generation_config=GenerationConfig(num_images_per_prompt=1).model_dump(),
        device="cpu",
    )
    metrics = evaluator.evaluate(PrecomputedImageModel(device="cpu"))

    assert metrics["score"] == pytest.approx(0.5)
    assert metrics["composition_score"] == pytest.approx(0.5)
    assert metrics["reasoning_score"] is None
    assert metrics["num_questions"] == 2
    assert metrics["valid_answer_rate"] == 1.0
    assert metrics["is_partial_evaluation"] is True
    questions = evaluator.artifact_records["questions.jsonl"]
    assert [record["score"] for record in questions] == [1, 0]


def test_aggregator_uses_equal_dimension_weighting_and_tracks_invalid_answers():
    def evaluation(sample_id, prompt_id, dimension, score, valid):
        checklist = [
            {"question": "q", "score": 1 if valid else None, "cached": valid}
        ]
        return SampleEvaluation(
            sample_id=sample_id,
            generation=GenerationResult(images=[]),
            metadata={
                "prompt_id": prompt_id,
                "dimension": dimension,
                "image_score": score,
                "checklist": checklist,
            },
        )

    metrics = aggregate_corebench_results(
        [
            evaluation("a", "p1", "C-MI", 0.5, True),
            evaluation("b", "p2", "C-MI", 1.0, True),
            evaluation("c", "p3", "R-CR", 0.25, False),
        ]
    )

    assert metrics["dimension_scores"] == {
        "C-MI": pytest.approx(0.75),
        "R-CR": pytest.approx(0.25),
    }
    assert metrics["score"] == pytest.approx(0.5)
    assert metrics["composition_score"] == pytest.approx(0.75)
    assert metrics["reasoning_score"] == pytest.approx(0.25)
    assert metrics["num_invalid_answers"] == 1
    assert metrics["cached_answer_rate"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("yes", ("yes", 1)), ("</think>\n\nno", ("no", 0)), ("maybe", (None, None))],
)
def test_binary_answer_parser(text, expected):
    assert parse_binary_answer(text) == expected
