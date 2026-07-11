from typing import Any

import PIL.Image
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # which device to run the model on
    # e.g., "cpu", "cuda", "cuda:0"
    device: str = "cuda"

    # random seed for reproducibility
    seed: int | None = None


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # which device to run the evaluator on
    # e.g., "cpu", "cuda", "cuda:0"
    device: str = "cuda"

    # random seed for reproducibility
    seed: int | None = None

    # (Optional) use accelerator to speed up inference and evaluation, if applicable.
    accelerator: Accelerator | None = None

    # if set, store every generated sample to the specified directory
    # if None, do not store any generated samples
    sample_dir: str | None = None

    # Reuse complete samples previously written to ``sample_dir``.
    resume: bool = False


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = ""
    negative_prompt: str | None = None
    steps: int = 50
    guidance_scale: float = 7.5
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    num_images_per_prompt: int = 1

    # Model-specific generation arguments.  Keeping these in a dedicated
    # mapping lets adapters support new pipelines without changing this core
    # schema for every upstream release.
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: list[PIL.Image.Image]
    debug_info: dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(GenerationConfig):
    """Stable model-adapter request contract.

    ``GenerationConfig`` remains the backwards-compatible benchmark-facing
    type.  New adapters may use ``sample_id`` to preserve identity across
    batching and retries.
    """

    sample_id: str | None = None


class GenerationOutput(GenerationResult):
    """Stable model-adapter output contract."""

    sample_id: str | None = None
    error: str | None = None
