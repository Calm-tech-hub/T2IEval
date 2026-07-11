# T2I-CoReBench evaluator

`t2i_corebench` is a framework-native evaluator for the official
[T2I-CoReBench](https://github.com/KlingAIResearch/T2I-CoReBench) benchmark.
It reads the official checklist JSON, obtains generated or existing images,
uses a local Qwen MLLM for binary visual questions, and writes unified T2IEval
artifacts.

## What is implemented

- All 12 official dimensions: `C-MI`, `C-MA`, `C-MR`, `C-TR`, `R-LR`,
  `R-BR`, `R-HR`, `R-PR`, `R-GR`, `R-AR`, `R-CR`, and `R-RR`.
- Any subset of dimensions through YAML.
- Diffusers generation, including Qwen-Image through the common model adapter.
- Existing-image evaluation through the generic `precomputed` model adapter.
- Official-style checklist questions and strict yes/no parsing.
- Per-question, per-image, per-dimension, Composition, Reasoning, and overall
  scores.
- Question cache, invalid-answer accounting, image resume, and structured
  output.

## Local judge environment

The project lock uses `transformers==4.57.3`, which supports Qwen-Image and has
also been verified with the existing GenAI-Bench CLIP-FlanT5 scorer. Qwen3.5
still uses an isolated vLLM environment because its serving stack has separate
Torch/CUDA requirements. The evaluator launches that judge only after the
generation model has been unloaded.

Create it once:

```bash
uv venv --python 3.13 /root/autodl-tmp/corebench-judge/.venv
uv pip install \
  --python /root/autodl-tmp/corebench-judge/.venv/bin/python \
  vllm qwen-vl-utils transformers
```

The YAML field `judge_python` must point to that environment's Python binary.
The model stays local; no image or prompt is sent to an external service.

## Generate Qwen-Image images and evaluate them

Start with `num_prompts: 2` in the YAML, then increase it after the smoke run.

```bash
HF_HOME=/root/autodl-tmp/hf_cache \
t2i-eval -e t2i_corebench \
  -f examples/t2i_corebench/run_t2i_corebench_qwen_image.yaml
```

The loader expands every prompt into one sample per image. This keeps Qwen-Image
generation memory bounded and makes image-level resume deterministic. With four
images, seeds are `seed`, `seed+1`, `seed+2`, and `seed+3`.

## Evaluate existing images

Use the official directory convention:

```text
<image_dir>/
├── C-MI/
│   ├── C-MI-001-0.png
│   └── C-MI-001-1.png
└── R-CR/
    └── R-CR-001-0.png
```

Set `image_dir` in `examples/t2i_corebench/run_t2i_corebench_existing_images.yaml`, then run:

```bash
t2i-eval -e t2i_corebench \
  -f examples/t2i_corebench/run_t2i_corebench_existing_images.yaml
```

`strict_images: true` rejects incomplete prompt groups instead of silently
changing the benchmark sample count.

## Important evaluator arguments

| Argument | Meaning | Default |
| --- | --- | --- |
| `data_dir` | Local directory containing the 12 official JSON files | download official HF dataset |
| `dimensions` | `all`, a comma-separated string, or a YAML list | `all` |
| `num_prompts` | Maximum prompts per selected dimension | all 90 |
| `image_dir` | Existing official-style image root | generate images |
| `strict_images` | Require the configured image count for each prompt | `true` |
| `run_mode` | `evaluate` or `generate_only` | `evaluate` |
| `judge_model` | Local Qwen checkpoint | `Qwen/Qwen3.5-9B` |
| `judge_python` | Python executable containing vLLM | current Python |
| `judge_cache_path` | Persistent question-result JSONL | run artifact cache when images are saved |
| `judge_batch_size` | vLLM maximum concurrent sequences | `64` |
| `judge_max_rounds` | Retries for empty thinking output | `3` |

## Output and score definitions

Every checklist answer is `1` for yes, `0` for no, or invalid when no binary
answer can be parsed. Invalid answers are reported and excluded from the image
mean, matching the official code's valid-score behavior.

```text
question scores -> image_score -> dimension score
dimension scores -> composition_score / reasoning_score
available dimension scores -> score
```

For a partial run, `score` is the equal-weight mean of the selected dimensions
and `is_partial_evaluation` is true. It must not be presented as the full
12-dimension leaderboard score.

The run directory includes:

- `metrics.json`: summary metrics and coverage.
- `samples.jsonl`: image-level records.
- `questions.jsonl`: every question, answer, score, cache state, and error.
- `judge_cache.jsonl`: reusable local Qwen answers when image saving/resume is enabled.
- `config.json` and `environment.json`: reproduction information.

## Reused framework components

- `BenchmarkSample` and `SampleEvaluation` for the internal contract.
- `DiffusersModel` for Qwen-Image and other compatible generators.
- `PrecomputedImageModel` for existing images.
- `SimpleEvaluator` for loading, generation, postprocessing, and aggregation.
- `ArtifactWriter` for unified output and errors.

No T2I-CoReBench-specific branch is added to the CLI.

## Differences from the official evaluation

The official main-paper evaluator is Gemini 2.5 Flash. This implementation
defaults to the officially supported open-source Qwen3.5-9B evaluator. It ports
the official system prompt, temperature, repetition penalty, multi-round retry,
and checklist aggregation logic, but runs vLLM in an isolated subprocess.

Generation seeds are explicit per image rather than relying on one process-wide
random-number stream. This improves resume and distributed reproducibility but
means newly generated pixels are not expected to duplicate the official image
files exactly.

## Limitations

- Qwen3.5-9B and Gemini 2.5 Flash can disagree, especially for small objects,
  rendered text, negation, and complex relations.
- Compare results only with the same judge model, prompt template, generated
  images, and inference settings.
- A full four-image run requires about 4,320 images and roughly 54,144 visual
  questions, so partial dimensions are useful for an engineering demonstration.
- Quantization, vLLM version, Qwen version, and invalid-answer rate can change
  results. Always report `valid_answer_rate` with the score.
