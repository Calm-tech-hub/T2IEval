# GenAI-Bench Parameters

GenAI-Bench evaluates text-image alignment and skill/tag-level performance.
In the exam package, this page describes the target behavior and expected configuration for the evaluator that candidates need to implement in `T2IEval`.

## Evaluator args (`-E genaibench:key=value`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `Vertsineu/geneaibench` | Hugging Face dataset path. |
| `dataset_config_name` | str | `image_1600` | Dataset subset config. Supported values in code: `image_1600`, `image_527`. |
| `split` | str | `test_v1` | Dataset split. |
| `num_samples` | int \| null | `null` | Limit number of prompts for quick tests/debugging. |
| `score_batch_size` | int | `16` | Batch size used by GenAI-Bench scorer postprocessor. |
| `seed` | int \| null | `null` | Evaluator-level seed used for sampling/shuffling when applicable. |
| `device` | str | `cuda` | Device used by evaluator/scoring stage. |
| `sample_dir` | str \| null | `null` | If set, save generated samples for inspection. |

## Per-eval generation overrides (`-G genaibench:key=value`)

Default template comes from `GenAIBenchGenerationConfig`.

| Key | Default | Notes |
| --- | --- | --- |
| `steps` | `50` | Diffusion steps. Lower values speed up smoke tests. |
| `guidance_scale` | `9.0` | CFG scale used by default for this benchmark. |
| `width`, `height` | `512`, `512` | Default image size for GenAI-Bench runs. |
| `seed` | `42` | Generation seed inside generation config. |
| `num_images_per_prompt` | `1` | Number of generated images per prompt. |
| `negative_prompt` | `null` | Optional prompt suppression string. |

## Sample command

```bash
uv run t2i-eval \
  -m diffusers \
  -a pretrained=runwayml/stable-diffusion-v1-5 \
  -e genaibench -E dataset_config_name=image_527,num_samples=32,score_batch_size=8 \
  -G steps=30,guidance_scale=8.0 \
  -o results_genaibench
```

This command is the intended target behavior for the implementation task. The exam package does not ship with a full GenAI-Bench implementation.
