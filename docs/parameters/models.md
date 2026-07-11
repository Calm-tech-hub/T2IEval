# Model Registry Parameters

Each CLI run picks one model via `-m/--model`. This document lists supported registry keys and their initialization kwargs (`-a/--model-args key=value`). Use commas to separate multiple `key=value` pairs; flags can be repeated to override later values.

## `diffusers`

Backed by `src/t2i_eval/model/diffusers_model.py`. Wraps Hugging Face Diffusers pipelines.

| Argument | Type | Required? | Description | Example |
| --- | --- | --- | --- | --- |
| `pretrained` | str | ✅ | HF repo ID or local path for the pipeline | `pretrained=runwayml/stable-diffusion-v1-5` |
| `pipeline` | str | ➖ | Override the Diffusers pipeline class name (e.g., custom wrapper). Requires the class to be importable. | `pipeline=ZImagePipeline` |
| `dtype` | str | ➖ (defaults to `float16`) | Torch dtype for weights (`float16`, `bfloat16`, `float32`) | `dtype=bfloat16` |
| `disable_safety_checker` | bool-ish str | ➖ (default `false`) | Set to `true` to skip Diffusers safety checker | `disable_safety_checker=true` |
| `device` | str | ➖ (default `cuda`) | Target device string passed to the model | `device=cuda:1` |
| `trust_remote_code` | bool-ish str | ➖ | Enable if loading custom pipelines | `trust_remote_code=true` |
| `revision` | str | ➖ | Specific git revision/tag for the repo | `revision=fp16` |

Example commands:

```bash
# SD 1.5 baseline
uv run t2i-eval -m diffusers \
  -a pretrained=runwayml/stable-diffusion-v1-5,dtype=float16,disable_safety_checker=true \
  ...

# SDXL with BF16 weights
uv run t2i-eval -m diffusers \
  -a pretrained=stabilityai/stable-diffusion-xl-base-1.0,dtype=bfloat16 \
  ...
```

> **Notes**
> - All `-a` values are parsed as strings; booleans use `true/false`.
> - Any extra key not listed here will raise `extra fields not permitted` because `ModelConfig` uses `extra="forbid"`.

## Adding new models

See `docs/extending.md` for instructions on registering additional models. Once registered, document their arguments here.
