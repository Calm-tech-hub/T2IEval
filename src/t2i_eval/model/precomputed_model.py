from pathlib import Path

from PIL import Image

from ..core.model import BaseModel
from ..core.registry import register_model
from ..core.schema import GenerationConfig, GenerationResult, ModelConfig


class PrecomputedConfig(ModelConfig):
    """Configuration for loading images produced by an earlier run."""

    image_mode: str = "RGB"


@register_model("precomputed")
class PrecomputedImageModel(BaseModel):
    """Treat existing image files as a generation-model output.

    Benchmark loaders put one or more absolute paths in
    ``GenerationConfig.extra_kwargs['image_paths']``.  Keeping this behavior in
    a model adapter lets every evaluator reuse the normal generation pipeline.
    """

    supports_precomputed_images = True

    def __init__(self, **kwargs):
        self.config = PrecomputedConfig(**kwargs)
        self._loaded = False

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, config: GenerationConfig) -> GenerationResult:
        if not self._loaded:
            self.load()

        raw_paths = config.extra_kwargs.get("image_paths")
        if raw_paths is None:
            single_path = config.extra_kwargs.get("image_path")
            raw_paths = [single_path] if single_path else []
        if isinstance(raw_paths, (str, Path)):
            raw_paths = [raw_paths]

        paths = [Path(path).expanduser().resolve() for path in raw_paths]
        if not paths:
            raise ValueError(
                "PrecomputedImageModel requires `image_path` or `image_paths` "
                "in generation_config.extra_kwargs."
            )

        images = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Precomputed image not found: {path}")
            with Image.open(path) as image:
                images.append(image.convert(self.config.image_mode).copy())

        return GenerationResult(
            images=images,
            debug_info={"source_paths": [str(path) for path in paths]},
        )


__all__ = ["PrecomputedImageModel"]
