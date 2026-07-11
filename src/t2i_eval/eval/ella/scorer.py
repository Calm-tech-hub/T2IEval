import gc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from PIL import Image


class VQAScorer(Protocol):
    """Minimal interface required by the DPG-Bench postprocessor."""

    def answer(self, image: Image.Image, question: str) -> str: ...


class MPlugVQAScorer:
    """ModelScope mPLUG adapter used by the official DPG-Bench evaluator."""

    def __init__(self, model_id: str, device: str):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            Tasks.visual_question_answering,
            model=model_id,
            device=device,
        )

    def answer(self, image: Image.Image, question: str) -> str:
        result = self.pipeline({"image": image, "question": question})
        text: Any = result.get("text", "") if isinstance(result, Mapping) else result
        if isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
            text = text[0] if text else ""
        return str(text).strip().lower()

    def close(self) -> None:
        self.pipeline = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def apply_question_dependencies(
    raw_scores: dict[int, float],
    questions: list[dict[str, Any]],
    missing_dependency_policy: str = "zero",
) -> tuple[dict[int, float], dict[int, bool], list[dict[str, int]]]:
    """Apply DPG parent-question constraints in question order.

    The official implementation mutates scores in insertion order, so children see
    already-adjusted parent scores. Self-dependencies are effectively a no-op. A
    missing parent can occur in the published HF conversion; the default policy
    turns the dependent proposition into zero instead of failing the full run.
    """

    if missing_dependency_policy not in {"zero", "ignore", "error"}:
        raise ValueError(
            "missing_dependency_policy must be one of: zero, ignore, error"
        )

    adjusted_scores = dict(raw_scores)
    validity: dict[int, bool] = {}
    missing_dependencies: list[dict[str, int]] = []

    for question in questions:
        qid = int(question["qid"])
        parent_failed = False

        for parent_id in question.get("dependency", []):
            parent_id = int(parent_id)
            if parent_id == 0 or parent_id == qid:
                continue

            if parent_id not in adjusted_scores:
                missing_dependencies.append({"qid": qid, "parent_qid": parent_id})
                if missing_dependency_policy == "error":
                    raise ValueError(
                        f"Question {qid} references missing parent {parent_id}."
                    )
                if missing_dependency_policy == "zero":
                    parent_failed = True
                continue

            if adjusted_scores[parent_id] == 0.0:
                parent_failed = True

        if parent_failed:
            adjusted_scores[qid] = 0.0
            validity[qid] = False
        else:
            validity[qid] = True

    return adjusted_scores, validity, missing_dependencies
