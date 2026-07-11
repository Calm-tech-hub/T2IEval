# GenEval Parameters

GenEval evaluates logical/attribute correctness of generated images. The evaluator config lives in `src/t2i_eval/eval/geneval/evaluator.py` (class `GenevalConfig`). Accepted CLI/config keys for `-E`/`gen_args` are listed below.

## Evaluator args (`-E geneval:key=value`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `Vertsineu/geneval` | HF dataset to load prompts/metadata. |
| `split` | str | `validation` | Dataset split. |
| `category` | str | `all` | Filter prompts by category (`single_object`, `two_object`, `counting`, `colors`, `position`, `color_attr`). |
| `seed` | int | `42` | Controls sampling + evaluator RNG. |
| `num_samples` | int \| null | `null` | Cap number of prompts (after category filtering). |
| `detector_model_id` | str | `facebook/mask2former-swin-small-coco-instance` | Detector for post-processing. |
| `detector_local_files_only` | bool-ish str | `false` | Force local HF cache. |
| `sample_dir` | str \| null | `null` | If set, save generated samples for inspection under this directory. |

## Per-eval generation overrides (`-G geneval:key=value`)

| Key | Default | Notes |
| --- | --- | --- |
| `steps` | 50 (GenevalGenerationConfig) | Override to speed up experiments; keep >=25 for stability. |
| `guidance_scale` | 9.0 | Higher CFG improves attribute adherence but may reduce diversity. |
| `num_images_per_prompt` | 4 | GenEval checks multiple images per prompt; ensure you have enough GPU memory. |
| `negative_prompt`, `width`, `height` | `null` | Optional; use to control aesthetics. |

## Sample command

```bash
uv run t2i-eval \
  -m diffusers \
  -a pretrained=runwayml/stable-diffusion-v1-5 \
  -g steps=20,seed=123 \
  -e geneval -E num_samples=32,category=colors -G num_images_per_prompt=4 \
  -o results_geneval
```
