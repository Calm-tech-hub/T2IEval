import gc
import importlib
import importlib.metadata
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from tqdm.auto import tqdm


class GenAIBenchScorer(Protocol):
    def score_pairs(
        self,
        images: list[Image.Image],
        texts: list[str],
        batch_size: int,
    ) -> list[float]: ...


def _load_clip_t5_module():
    """Load only t2v-metrics CLIP-FlanT5 implementation."""
    distribution = importlib.metadata.distribution("t2v-metrics")
    root = Path(str(distribution.locate_file("t2v_metrics")))
    namespaces = {
        "t2v_metrics": root,
        "t2v_metrics.models": root / "models",
        "t2v_metrics.models.vqascore_models": root
        / "models"
        / "vqascore_models",
    }
    created_names = []
    for name, path in namespaces.items():
        if name in sys.modules:
            continue
        package = types.ModuleType(name)
        package.__package__ = name
        setattr(package, "__path__", [str(path)])
        sys.modules[name] = package
        created_names.append(name)

    try:
        return importlib.import_module(
            "t2v_metrics.models.vqascore_models.clip_t5_model"
        )
    except Exception:
        # Do not leave synthetic namespace packages behind after a failed
        # optional import; they would poison a later regular t2v_metrics import.
        for name in reversed(created_names):
            sys.modules.pop(name, None)
        raise


class ClipFlanT5Scorer:
    """Adapter for the paper-reproduction VQAScore implementation."""

    def __init__(
        self,
        model_name: str,
        device: str,
        cache_dir: str | None = None,
        required_version: str = "1.1",
        question_template: str | None = None,
        answer_template: str | None = None,
    ):
        try:
            installed_version = importlib.metadata.version("t2v-metrics")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"GenAI-Bench requires t2v-metrics=={required_version}. "
                "Run `uv sync` to install the locked project environment."
            ) from exc

        if installed_version != required_version:
            raise RuntimeError(
                "GenAI-Bench paper reproduction requires t2v-metrics=="
                f"{required_version}, but {installed_version} is installed."
            )

        try:
            clip_t5 = _load_clip_t5_module()
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the CLIP-FlanT5 implementation from "
                f"t2v-metrics {required_version}. Ensure its runtime "
                "dependencies are installed."
            ) from exc

        if model_name not in clip_t5.CLIP_T5_MODELS:
            raise ValueError(
                f"Unsupported GenAI-Bench scorer {model_name!r}; expected a "
                "CLIP-FlanT5 model from t2v-metrics."
            )

        kwargs = {"model_name": model_name, "device": device}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        self.score_model: Any = clip_t5.CLIPT5Model(**kwargs)
        self.question_template = question_template
        self.answer_template = answer_template

    def score_pairs(
        self,
        images: list[Image.Image],
        texts: list[str],
        batch_size: int,
    ) -> list[float]:
        if len(images) != len(texts):
            raise ValueError("GenAI-Bench images and texts must have equal length.")
        if not images:
            return []

        if batch_size <= 0:
            raise ValueError("GenAI-Bench score batch size must be positive.")

        with tempfile.TemporaryDirectory(prefix="t2i_eval_genaibench_") as temp_dir:
            root = Path(temp_dir)
            image_paths = []
            for index, image in enumerate(images):
                image_path = root / f"{index:06d}.png"
                image.save(image_path)
                image_paths.append(str(image_path))

            score_kwargs = {}
            if self.question_template is not None:
                score_kwargs["question_template"] = self.question_template
            if self.answer_template is not None:
                score_kwargs["answer_template"] = self.answer_template

            scores: list[float] = []
            starts = range(0, len(image_paths), batch_size)
            for start in tqdm(
                starts,
                total=(len(image_paths) + batch_size - 1) // batch_size,
                desc="Evaluating GenAI-Bench",
                unit="batch",
            ):
                end = min(start + batch_size, len(image_paths))
                batch_scores = self.score_model.forward(
                    image_paths[start:end],
                    texts[start:end],
                    **score_kwargs,
                )
                values = batch_scores.detach().float().cpu().reshape(-1)
                scores.extend(float(value) for value in values)
            return scores

    def close(self) -> None:
        self.score_model = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
