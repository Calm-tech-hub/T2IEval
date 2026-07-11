import gc
import inspect
import time

import torch
from accelerate import Accelerator

from ..core.model import BaseModel
from ..core.registry import register_model
from ..core.schema import GenerationConfig, GenerationResult, ModelConfig


class DiffusersConfig(ModelConfig):
    pretrained: str
    pipeline: str = "DiffusionPipeline"

    dtype: str = "float16"
    variant: str | None = None
    revision: str = "main"
    disable_safety_checker: bool = False
    scheduler: str | None = None

    # LoRA Configuration
    lora_path: str | None = None
    lora_weight_name: str | None = None
    lora_scale: float = 1.0

    # Device Configuration
    device_map: str | None = None

    # Memory Optimization
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False

    # Optional inference/runtime knobs for benchmark alignment.
    enable_attention_slicing: bool = False
    attention_slice_size: str | int | None = None
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_xformers_memory_efficient_attention: bool = False

    # Optional prompt/pipeline arguments for SDXL-like pipelines.
    mirror_prompt_to_prompt_2: bool = False
    mirror_negative_prompt_to_negative_prompt_2: bool = False
    clip_skip: int | None = None
    max_sequence_length: int | None = None
    output_type: str | None = None


@register_model("diffusers")
class DiffusersModel(BaseModel):
    def __init__(self, **kwargs):
        # Declarative parsing: "args are first kwargs -> Config class parses & fills defaults"
        # We assume kwargs comes from CLI (dict of strings) or direct python usage
        self.config = DiffusersConfig(**kwargs)

        self._loaded = False
        self.pipeline = None
        self.accelerator = None

    def _pipeline_supports(self, name: str) -> bool:
        assert self.pipeline is not None, "Pipeline must be initialized before inspection."
        try:
            signature = inspect.signature(self.pipeline.__call__)
        except (TypeError, ValueError):
            return False
        return name in signature.parameters

    def enable_accelerator(self, accelerator: Accelerator):
        if self._loaded:
            raise RuntimeError(
                "Cannot enable accelerator after the model has been loaded."
            )

        self.accelerator = accelerator

    def load(self):
        assert self.pipeline is None, "Model has already been loaded."

        import diffusers

        dtype = self.config.dtype.strip()
        dtype = getattr(torch, dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Unsupported dtype '{self.config.dtype}'. "
                "Use a valid `torch` dtype attribute name such as "
                "'float16', 'float32', or 'bfloat16'."
            )

        pipeline = getattr(diffusers, self.config.pipeline, None)
        if pipeline is None:
            raise ValueError(
                f"Unsupported pipeline class '{self.config.pipeline}'. "
                "Please provide a valid class name from `diffusers`."
            )
        if not hasattr(pipeline, "from_pretrained"):
            raise TypeError(
                f"Pipeline class '{self.config.pipeline}' does not support "
                "`from_pretrained`."
            )

        self.pipeline = pipeline.from_pretrained(
            self.config.pretrained,
            device_map=self.config.device_map,
            torch_dtype=dtype,
            variant=self.config.variant,
            revision=self.config.revision,
        )

        if self.config.scheduler is not None and self.pipeline is not None:
            scheduler = getattr(diffusers, self.config.scheduler, None)
            if scheduler is None:
                raise ValueError(
                    f"Unsupported scheduler class '{self.config.scheduler}'. "
                    "Please provide a valid scheduler class name from `diffusers`."
                )
            if not hasattr(scheduler, "from_config"):
                raise TypeError(
                    f"Scheduler class '{self.config.scheduler}' does not support "
                    "`from_config`."
                )
            self.pipeline.scheduler = scheduler.from_config(
                self.pipeline.scheduler.config
            )

        if self.config.disable_safety_checker and self.pipeline is not None:
            has_safety_checker = hasattr(self.pipeline, "safety_checker")
            if has_safety_checker:
                # Disable NSFW filtering for benchmark reproducibility experiments.
                self.pipeline.safety_checker = None
            # Some pipelines (e.g., SDXL) do not expect `requires_safety_checker`
            # in their component config; only set it for pipelines that actually
            # expose a safety checker component.
            if has_safety_checker and hasattr(self.pipeline, "register_to_config"):
                self.pipeline.register_to_config(requires_safety_checker=False)

        if self.config.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        elif self.config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        elif not self.config.device_map:
            device = self.accelerator.device if self.accelerator else self.config.device
            self.pipeline = self.pipeline.to(device)

        if self.config.enable_attention_slicing and hasattr(
            self.pipeline, "enable_attention_slicing"
        ):
            self.pipeline.enable_attention_slicing(self.config.attention_slice_size)
        if self.config.enable_vae_slicing and hasattr(
            self.pipeline, "enable_vae_slicing"
        ):
            self.pipeline.enable_vae_slicing()
        if self.config.enable_vae_tiling and hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
        if self.config.enable_xformers_memory_efficient_attention and hasattr(
            self.pipeline, "enable_xformers_memory_efficient_attention"
        ):
            self.pipeline.enable_xformers_memory_efficient_attention()

        if self.config.lora_path:
            self.pipeline.load_lora_weights(
                self.config.lora_path, weight_name=self.config.lora_weight_name
            )
            if self.config.lora_scale != 1.0:
                self.pipeline.fuse_lora(lora_scale=self.config.lora_scale)

        self._loaded = True

    def unload(self):
        if not self._loaded:
            return

        # 1. Synchronize all processes in distributed settings
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        # 2. Delete the pipeline object to remove Python references
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        # 3. Force Python garbage collection
        gc.collect()

        # 4. Clear PyTorch's CUDA cache to actually release VRAM back to the OS
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # 5. Reset the loading state
        self._loaded = False

    def generate(self, config: GenerationConfig) -> GenerationResult:
        if not self._loaded:
            self.load()

        assert self.pipeline is not None, "Model has not been loaded."

        device = self.accelerator.device if self.accelerator else self.config.device
        generator = None
        if config.seed is not None:
            if config.num_images_per_prompt > 1:
                generator = [
                    torch.Generator(device=device).manual_seed(config.seed + i)
                    for i in range(config.num_images_per_prompt)
                ]
            else:
                generator = torch.Generator(device=device).manual_seed(config.seed)

        start_time = time.time()
        pipeline_kwargs = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "num_inference_steps": config.steps,
            "guidance_scale": config.guidance_scale,
            "width": config.width,
            "height": config.height,
            "num_images_per_prompt": config.num_images_per_prompt,
            "generator": generator,
        }
        if self.config.mirror_prompt_to_prompt_2 and self._pipeline_supports("prompt_2"):
            pipeline_kwargs["prompt_2"] = config.prompt
        if (
            self.config.mirror_negative_prompt_to_negative_prompt_2
            and self._pipeline_supports("negative_prompt_2")
        ):
            pipeline_kwargs["negative_prompt_2"] = config.negative_prompt
        if self.config.clip_skip is not None and self._pipeline_supports("clip_skip"):
            pipeline_kwargs["clip_skip"] = self.config.clip_skip
        if self.config.max_sequence_length is not None and self._pipeline_supports(
            "max_sequence_length"
        ):
            pipeline_kwargs["max_sequence_length"] = self.config.max_sequence_length
        if self.config.output_type is not None and self._pipeline_supports("output_type"):
            pipeline_kwargs["output_type"] = self.config.output_type

        images = self.pipeline(
            **pipeline_kwargs,
        ).images  # pyright: ignore[reportCallIssue]
        latency = time.time() - start_time

        return GenerationResult(images=images, debug_info={"latency": latency})
