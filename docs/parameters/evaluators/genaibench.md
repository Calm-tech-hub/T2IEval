# GenAI-Bench Parameters

GenAI-Bench evaluates text-image alignment with CLIP-FlanT5 VQAScore and
reports both an overall mean and skill/tag-level means. The evaluator is
registered as `genaibench` and runs through the same loader, generation,
postprocessor, aggregator, and JSON-output path as `geneval`.

## Evaluator args (`-E genaibench:key=value`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `Vertsineu/geneaibench` | Hugging Face dataset path. |
| `dataset_config_name` | str | `image_1600` | Dataset subset config. Supported values in code: `image_1600`, `image_527`. |
| `split` | str | `test_v1` | Dataset split. |
| `num_samples` | int \| null | `null` | Limit number of prompts for quick tests/debugging. |
| `score_batch_size` | int | `16` | Batch size used by GenAI-Bench scorer postprocessor. |
| `scorer_model` | str | `clip-flant5-xxl` | CLIP-FlanT5 checkpoint key. Use `clip-flant5-xl` only for development smoke tests. |
| `scorer_version` | str | `1.1` | Required `t2v-metrics` package version. Runtime validation rejects a mismatch. |
| `scorer_cache_dir` | str \| null | `null` | Cache directory passed to the main CLIP-FlanT5 checkpoint loader. Also set `HF_HOME` so tokenizer/vision-tower lookups use the same cache root. |
| `question_template` | str \| null | `null` | Optional VQAScore question override. Leave unset for paper reproduction. |
| `answer_template` | str \| null | `null` | Optional VQAScore answer override. Leave unset for paper reproduction. |
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

## Ready-made configurations

| File | Purpose |
| --- | --- |
| `examples/genaibench/run_genaibench_smoke.yaml` | One SD1.5 image, two diffusion steps; validates plumbing only. |
| `examples/genaibench/run_genaibench_sd21.yaml` | Full 1,600-prompt SD2.1 configuration. |
| `examples/genaibench/run_genaibench_sdxl.yaml` | Full 1,600-prompt SDXL configuration at 1024×1024. |

Before running, point all Hugging Face consumers at the shared cache:

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
```

One-sample end-to-end command:

```bash
uv run --frozen t2i-eval -f examples/genaibench/run_genaibench_smoke.yaml --fail-fast
```

## Equivalent CLI example

```bash
uv run t2i-eval \
  -m diffusers \
  -a pretrained=stable-diffusion-v1-5/stable-diffusion-v1-5,dtype=float16,disable_safety_checker=true \
  -e genaibench \
  -E genaibench:dataset_config_name=image_1600,num_samples=1,score_batch_size=1 \
  -G genaibench:steps=2,guidance_scale=7.5,seed=42 \
  -o results_genaibench
```

The smoke configuration deliberately changes the generation model/steps and
therefore must not be used for paper-result comparison. Use one of the two
full YAML files for reported benchmark numbers.

## Output

The CLI writes `results_genaibench_diffusers.json`. Its `metrics` object has:

- `score`: mean VQAScore over tagged prompts;
- `task_scores`: mean VQAScore grouped by every dataset skill plus `all`.

Implementation and reproduction details are recorded in
[`docs/genaibench_implementation_notes.md`](../../genaibench_implementation_notes.md).
