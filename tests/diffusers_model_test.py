from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from t2i_eval.core.schema import GenerationConfig
from t2i_eval.model.diffusers_model import DiffusersModel

if __name__ == "__main__":
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=6))
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs])
    model = DiffusersModel(
        pretrained="runwayml/stable-diffusion-v1-5",
        dtype="float16",
        device_map=None,
    )
    model.enable_accelerator(accelerator)
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "a photo of a cat sitting on a windowsill",
        "a photo of a bowl of fruit on a table",
        "a photo of a city skyline at sunset",
        "a photo of a dog playing in the park",
        "a photo of a mountain landscape with a river",
    ]

    configs = [
        GenerationConfig(prompt=prompt, steps=10, seed=42, num_images_per_prompt=5)
        for prompt in prompts
    ]
    results = model.generate_batch(configs)
    for i, result in enumerate(results):
        print(f"Prompt: {prompts[i]}")
        print(
            f"Generated {len(result.images)} images with debug info: {result.debug_info}"
        )

    result = model.generate(
        GenerationConfig(
            prompt="a photo of an astronaut riding a horse on mars",
            steps=10,
            seed=42,
            num_images_per_prompt=5,
        )
    )
    print(f"Generated {len(result.images)} images with debug info: {result.debug_info}")

    # print memory usage before unload
    if torch.cuda.is_available():
        print(f"VRAM usage before unload: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model.unload()

    # print memory usage to verify that VRAM is released
    if torch.cuda.is_available():
        print(f"VRAM usage after unload: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    accelerator.end_training()
