from abc import ABC, abstractmethod

from accelerate import Accelerator
from tqdm.auto import tqdm

from .schema import GenerationConfig, GenerationResult


class BaseModel(ABC):
    # Indicates whether the model has been loaded into memory.
    _loaded: bool = False

    def __init__(self, **kwargs):
        """
        Initialize the model with given parameters, but do not load it into memory.

        Args:
            config (ModelConfig): The configuration for the model.
        """
        pass

    def enable_accelerator(self, accelerator: Accelerator):
        """
        (Optional) Enable accelerator-specific optimizations if applicable. This method can be overridden by subclasses to implement device-specific optimizations.

        Args:
            accelerator (Accelerator): The accelerator instance to use for optimizations.

        Warning: Calling this method after the model has been loaded may not apply optimizations correctly. It is recommended to call this method before loading the model.
        """
        raise NotImplementedError(
            "enable_accelerator method is not implemented for this model."
        )

    @abstractmethod
    def load(self):
        """
        Load the model into memory.
        """
        pass

    @abstractmethod
    def unload(self):
        """
        Unload the model from memory to free up resources.
        """
        pass

    @abstractmethod
    def generate(self, config: GenerationConfig) -> GenerationResult:
        """
        Generate an image based on the given configuration. If the model has not been loaded, it will automatically load it.

        Args:
            config (GenerationConfig): The configuration for image generation.

        Returns:
            GenerationResult: The result of the image generation.
        """
        pass

    def generate_batch(self, configs: list[GenerationConfig]) -> list[GenerationResult]:
        """
        Generate a batch of images based on the given configuration.

        Args:
            configs (list[GenerationConfig]): The list of configurations for generation.

        Returns:
            list[GenerationResult]: A list of generation results.
        """
        if not self._loaded:
            self.load()

        # naive implementation, can be overridden by models that support batch generation
        results = []
        accelerator = getattr(self, "accelerator", None)
        disable_progress = (
            accelerator is not None and not accelerator.is_main_process
        )
        for config in tqdm(
            configs, desc="Generating", unit="prompt", disable=disable_progress
        ):
            result = self.generate(config)
            results.append(result)
        return results
