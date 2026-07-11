from pathlib import Path

import pytest
from PIL import Image

from t2i_eval.core.benchmark import SampleEvaluation
from t2i_eval.core.registry import get_evaluator_class
from t2i_eval.core.schema import GenerationResult
from t2i_eval.eval.ella.evaluator import (
    EllaConfig,
    EllaGenerationConfig,
    EllaPostprocessor,
    aggregate_ella_results,
    load_ella_records,
)
from t2i_eval.eval.ella.scorer import apply_question_dependencies


class FakeScorer:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def answer(self, image: Image.Image, question: str) -> str:
        return self.answers[question]


def _evaluation(metadata: dict, image_count: int = 1) -> SampleEvaluation:
    return SampleEvaluation(
        sample_id=str(metadata.get("item_id", "sample")),
        prompt=str(metadata.get("prompt", "")),
        generation=GenerationResult(
            images=[Image.new("RGB", (8, 8)) for _ in range(image_count)]
        ),
        metadata=metadata,
    )


def _question(
    qid: int,
    text: str,
    dependency: list[int],
    broad: str = "entity",
    detailed: str = "whole",
) -> dict:
    return {
        "qid": qid,
        "question": text,
        "dependency": dependency,
        "category_broad": broad,
        "category_detailed": detailed,
        "tuple": f"{broad} - {detailed}",
    }


def test_ella_is_registered_and_has_benchmark_defaults():
    assert get_evaluator_class("ella") is not None
    config = EllaConfig()
    assert config.dataset_name == "Vertsineu/ella"
    assert config.generation_config.guidance_scale == 12.0
    assert config.generation_config.num_images_per_prompt == 4


def test_dependency_propagation_and_missing_parent_policy():
    questions = [
        _question(1, "root", [0]),
        _question(2, "child", [1]),
        _question(3, "missing", [99]),
    ]
    adjusted, validity, missing = apply_question_dependencies(
        {1: 0.0, 2: 1.0, 3: 1.0}, questions
    )

    assert adjusted == {1: 0.0, 2: 0.0, 3: 0.0}
    assert validity == {1: True, 2: False, 3: False}
    assert missing == [{"qid": 3, "parent_qid": 99}]


def test_postprocessor_keeps_raw_and_dependency_scores_separate():
    questions = [
        _question(1, "Is there a parent?", [0]),
        _question(2, "Is there a child?", [1], "attribute", "color"),
        _question(3, "Is there a room?", [0]),
    ]
    scorer = FakeScorer(
        {
            "Is there a parent?": "no",
            "Is there a child?": "yes",
            "Is there a room?": "yes",
        }
    )
    postprocessor = EllaPostprocessor(
        device="cpu",
        vqa_model_id="unused",
        scorer=scorer,
    )
    result = postprocessor(
        [_evaluation({"item_id": "one", "questions": questions})]
    )[0].metadata

    assert result["score"] == pytest.approx(1 / 3)
    child_metric = next(
        metric for metric in result["question_metrics"] if metric["qid"] == 2
    )
    assert child_metric["raw_score"] == 1.0
    assert child_metric["dependency_score"] == 0.0
    assert child_metric["valid"] is False


def test_aggregator_matches_official_last_image_category_behavior():
    metadata = {
        "score": 0.5,
        "num_images": 2,
        "missing_dependencies": [],
        "question_metrics": [
            {
                "image_index": 0,
                "raw_score": 0.0,
                "category_broad": "entity",
                "category_detailed": "whole",
            },
            {
                "image_index": 1,
                "raw_score": 1.0,
                "category_broad": "entity",
                "category_detailed": "whole",
            },
        ],
    }

    evaluation = _evaluation(metadata, image_count=2)
    official = aggregate_ella_results([evaluation], "official_last_image")
    all_images = aggregate_ella_results([evaluation], "all_images")

    assert official["score"] == 0.5
    assert official["task_scores"]["entity"] == 1.0
    assert all_images["task_scores"]["entity"] == 0.5


def test_csv_loader_groups_questions_and_keeps_first_data_row(tmp_path: Path):
    csv_path = tmp_path / "dpg.csv"
    csv_path.write_text(
        "item_id,text,proposition_id,dependency,category_broad,"
        "category_detailed,tuple,question_natural_language\n"
        'sample1,"a red cube",1,0,entity,whole,'
        '"entity - whole (cube)","Is there a cube?"\n'
        'sample1,"a red cube",2,1,attribute,color,'
        '"attribute - color (cube, red)","Is the cube red?"\n',
        encoding="utf-8",
    )

    records = load_ella_records(
        dataset_name="unused",
        split="validation",
        default_config=EllaGenerationConfig(seed=42),
        csv_path=str(csv_path),
    )

    assert len(records) == 1
    sample = records[0]
    assert sample.generation_config.prompt == "a red cube"
    assert [question["qid"] for question in sample.metadata["questions"]] == [1, 2]
