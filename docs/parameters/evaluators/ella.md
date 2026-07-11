# ELLA Parameters

ELLA measures alignment between prompts and generated images using a QA-style benchmark. This evaluator is implemented inside `T2IEval` and runs the ELLA DPG-Bench through the unified loader, generation, postprocessor, and aggregator lifecycle.

## Evaluator args

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `Vertsineu/ella` | HF dataset hosting prompts/questions. |
| `split` | str | `validation` | Dataset split. |
| `num_samples` | int \| null | `null` | Number of prompts to evaluate (after filtering). |
| `seed` | int | `1001` | Sampling RNG. |
| `accelerator` | auto | Passed via CLI when running with `accelerate`; normally leave unset. |
| `device` | str | `cuda` | Evaluation device; override for CPU runs. |
| `sample_dir` | str \| null | `null` | If set, save generated samples for inspection. |
| `csv_path` | str \| null | `null` | Optional path to the official `dpg_bench.csv`; otherwise use the HF dataset. |
| `vqa_model_id` | str | `damo/mplug_visual-question-answering_coco_large_en` | ModelScope model ID or a local model directory. |
| `missing_dependency_policy` | str | `zero` | Missing-parent behavior: `zero`, `ignore`, or `error`. |
| `category_image_policy` | str | `official_last_image` | Use official last-image category behavior or `all_images`. |

## Generation overrides

ELLA uses `EllaGenerationConfig` defaults:

| Key | Default | Notes |
| --- | --- | --- |
| `steps` | `50` | Diffusion steps. |
| `guidance_scale` | `12.0` | ELLA default guidance scale. |
| `width`, `height` | `512`, `512` | Default image size for ELLA. |
| `num_images_per_prompt` | `4` | Number of images generated per prompt. |
| `seed` | `null` | Optional generation seed; evaluator sampling seed is controlled by `-E seed`. |

## Sample command

```bash
uv run --no-sync t2i-eval \
  -m diffusers \
  -a pretrained=stable-diffusion-v1-5/stable-diffusion-v1-5 \
  -g steps=50,guidance_scale=12.0,seed=42,width=512,height=512,num_images_per_prompt=4 \
  -e ella -E split=validation \
  -o /root/autodl-tmp/exam/results_ella_sd15
```

This command runs the framework-native ELLA evaluator and writes `results_ella_diffusers.json`.

For more complex pipelines (e.g., multi-GPU), wrap with `accelerate launch` as described in README Section 5.
