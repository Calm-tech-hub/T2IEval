# T2IEval

Run text-to-image evaluation with a single command.

| [Quick Start](#0-first-contact) | [Examples](#1-reading-the-examples) | [Parameter Docs](#21-documentation-map) | [CLI Reference](#3-cli-quick-reference) | [Extending](docs/guides/extending.md) |
| --- | --- | --- | --- | --- |

If this is your first visit, run one short command first and keep scrolling. The rest of the page gradually shifts from runnable snippets to customization details, so the abstract parts stay grounded in something you have already executed.

---

## 0. First Contact

T2IEval is designed for quick iteration: run something small, confirm it works, then scale.
In this exam implementation, `geneval` is the provided reference evaluator and both
`genaibench` and `ella` are integrated behind the same CLI and result schema.
The optional `t2i_corebench` evaluator demonstrates the same interfaces on a
12-dimension composition-and-reasoning benchmark with a local Qwen judge.

### 0.1 Environment setup

```bash
unzip T2IEval-test.zip
cd T2IEval-test
uv sync
```

### 0.2 A quick first pass

```bash
uv run t2i-eval \
  -m diffusers \
  -a pretrained=runwayml/stable-diffusion-v1-5,dtype=float16,disable_safety_checker=true \
  -g steps=20,seed=42 \
  -e geneval -E num_samples=16 \
  -o results_quickstart
```

A result file will be written under a path like `results_quickstart/results_geneval_diffusers.json`.

### 0.3 A few ready-made runs

| Goal | Command |
| --- | --- |
| Minimal smoke test | `uv run t2i-eval -f examples/geneval/run_basic.yaml -o results_basic` |
| Config-file run | `uv run t2i-eval -f examples/geneval/run_multi.yaml -o results_multi` |
| Geneval suite | `uv run t2i-eval -f examples/geneval/run_suite.yaml -o results_suite` |
| GenAI-Bench one-sample smoke | `uv run t2i-eval -f examples/genaibench/run_genaibench_smoke.yaml` |
| GenAI-Bench SD2.1 paper run | `uv run t2i-eval -f examples/genaibench/run_genaibench_sd21.yaml` |
| GenAI-Bench SDXL paper run | `uv run t2i-eval -f examples/genaibench/run_genaibench_sdxl.yaml` |
| ELLA SD1.5 paper run | `uv run t2i-eval -f examples/ella/run_ella_sd15.yaml` |
| T2I-CoReBench Qwen-Image partial run | `uv run t2i-eval -f examples/t2i_corebench/run_t2i_corebench_qwen_image.yaml` |
| T2I-CoReBench existing images | `uv run t2i-eval -f examples/t2i_corebench/run_t2i_corebench_existing_images.yaml` |
| SDXL baseline | `uv run t2i-eval -f examples/geneval/run_sdxl.yaml -o results_sdxl` |
| Custom pipeline example | `uv run t2i-eval -f examples/geneval/run_zimage_turbo.yaml -o results_flux` |

Pick one or two rows that match your current goal, then come back to the next section to decode the parameters you just used.

---

## 1. Reading the Examples

With one or two runs completed, the command lines become much easier to read. The notes below focus on what each parameter changes.

### 1.1 Example A: Single Geneval run

Command:

```bash
uv run t2i-eval \
  -m diffusers \
  -a pretrained=runwayml/stable-diffusion-v1-5,dtype=float16,disable_safety_checker=true \
  -g steps=20,seed=42 \
  -e geneval -E num_samples=16 \
  -o results_quickstart
```

Parameters used in this example:

| Parameter | Meaning |
| --- | --- |
| `-m diffusers` | Use the Diffusers model backend. |
| `-a pretrained=...` | Choose which pretrained model to load. |
| `-a dtype=float16` | Set model precision to FP16. |
| `-a disable_safety_checker=true` | Disable safety checker in the pipeline. |
| `-g steps=20` | Set diffusion sampling steps. |
| `-g seed=42` | Fix RNG seed for reproducibility. |
| `-e geneval` | Add the `geneval` evaluator. |
| `-E num_samples=16` | Evaluate up to 16 prompts/samples for this evaluator. |
| `-o results_quickstart` | Write outputs to this directory. |

### 1.2 Example B: Config-file run

Command:

```bash
uv run t2i-eval -f examples/geneval/run_multi.yaml -o results_multi
```

Parameters used in this example:

| Parameter | Meaning |
| --- | --- |
| `-f examples/geneval/run_multi.yaml` | Load model, generation, and evaluator settings from YAML. |
| `-o results_multi` | Override output directory for this run. |

Related example files:

- [examples/geneval/run_basic.yaml](examples/geneval/run_basic.yaml)
- [examples/geneval/run_multi.yaml](examples/geneval/run_multi.yaml)
- [examples/geneval/run_suite.yaml](examples/geneval/run_suite.yaml)
- [examples/genaibench/run_genaibench_smoke.yaml](examples/genaibench/run_genaibench_smoke.yaml)
- [examples/genaibench/run_genaibench_sd21.yaml](examples/genaibench/run_genaibench_sd21.yaml)
- [examples/genaibench/run_genaibench_sdxl.yaml](examples/genaibench/run_genaibench_sdxl.yaml)
- [examples/ella/run_ella_sd15.yaml](examples/ella/run_ella_sd15.yaml)
- [examples/t2i_corebench/run_t2i_corebench_qwen_image.yaml](examples/t2i_corebench/run_t2i_corebench_qwen_image.yaml)
- [examples/t2i_corebench/run_t2i_corebench_existing_images.yaml](examples/t2i_corebench/run_t2i_corebench_existing_images.yaml)
- [examples/geneval/run_sdxl.yaml](examples/geneval/run_sdxl.yaml)
- [examples/geneval/run_zimage_turbo.yaml](examples/geneval/run_zimage_turbo.yaml)

### 1.3 Example C: Geneval suite config

Command:

```bash
uv run t2i-eval -f examples/geneval/run_suite.yaml -o results_suite
```

Parameters used in this example:

| Parameter | Meaning |
| --- | --- |
| `-f examples/geneval/run_suite.yaml` | Load a geneval-focused suite config. |
| `-o results_suite` | Override output directory for this run. |

### 1.4 Example D: Custom pipeline (Z-Image Turbo)

Command:

```bash
uv run t2i-eval \
  -m diffusers \
  -a pipeline=ZImagePipeline,pretrained=Tongyi-MAI/Z-Image-Turbo,dtype=bfloat16 \
  -g steps=8,seed=42,guidance_scale=0.0,height=1024,width=1024 \
  -e geneval -E num_samples=2,sample_dir=./samples \
  -o results_flux
```

Parameters used in this example:

| Parameter | Meaning |
| --- | --- |
| `-a pipeline=ZImagePipeline` | Use a custom pipeline class. |
| `-a pretrained=Tongyi-MAI/Z-Image-Turbo` | Load weights from this model repo. |
| `-a dtype=bfloat16` | Set precision to BF16. |
| `-g steps=8` | Use fewer steps for faster generation. |
| `-g guidance_scale=0.0` | Set CFG guidance scale. |
| `-g height=1024,width=1024` | Set output image size. |
| `-E sample_dir=./samples` | Save generated samples for inspection. |

Equivalent config-file example: [examples/geneval/run_zimage_turbo.yaml](examples/geneval/run_zimage_turbo.yaml)

### 1.5 Example E: Accelerate multi-GPU launch

Command:

```bash
uv run accelerate launch t2i-eval \
  -f examples/geneval/run_multi.yaml \
  -o results_accel \
  --fail-fast
```

Parameters used in this example:

| Parameter | Meaning |
| --- | --- |
| `accelerate launch` | Run the same CLI through Accelerate for multi-process/multi-GPU setup. |
| `-f examples/geneval/run_multi.yaml` | Use YAML as the evaluation plan. |
| `-o results_accel` | Output directory. |
| `--fail-fast` | Stop immediately when any evaluator fails. |

---

## 2. Make It Your Own

Once the examples feel familiar, this is where the project becomes configurable rather than prescriptive.

### 2.1 Documentation map

- Main parameter index: [docs/parameters/overview.md](docs/parameters/overview.md)
- Model parameters (`-m` / `-a`): [docs/parameters/models.md](docs/parameters/models.md)
- Generation parameters (`-g` / `-G`): [docs/parameters/generation.md](docs/parameters/generation.md)
- Evaluator parameters:
  - Geneval: [docs/parameters/evaluators/geneval.md](docs/parameters/evaluators/geneval.md)
  - Ella: [docs/parameters/evaluators/ella.md](docs/parameters/evaluators/ella.md)
  - GenAI-Bench: [docs/parameters/evaluators/genaibench.md](docs/parameters/evaluators/genaibench.md)
  - T2I-CoReBench: [docs/parameters/evaluators/t2i_corebench.md](docs/parameters/evaluators/t2i_corebench.md)
- Extend models or evaluators: [docs/guides/extending.md](docs/guides/extending.md)

Tip: if you only need a quick tweak, jump directly to the evaluator page you are running and scan `-E` / `-G` fields first.

### 2.2 CLI vs YAML mapping

| What you want to change | CLI syntax | YAML syntax | Documentation |
| --- | --- | --- | --- |
| Select model backend | `-m diffusers` | `model: diffusers` | [docs/parameters/models.md](docs/parameters/models.md) |
| Set model initialization args | `-a key=value` | `model_args.<key>: value` | [docs/parameters/models.md](docs/parameters/models.md) |
| Set global generation args | `-g key=value` | `generation.<key>: value` | [docs/parameters/generation.md](docs/parameters/generation.md) |
| Add an evaluation task | `-e geneval` | `evaluations.geneval` | [docs/parameters/overview.md](docs/parameters/overview.md) |
| Set evaluator-specific args | `-E key=value` or `-E name:key=value` | `evaluations.<name>.eval_args.<key>: value` | corresponding evaluator doc |
| Set evaluator-specific generation overrides | `-G key=value` or `-G name:key=value` | `evaluations.<name>.gen_args.<key>: value` | [docs/parameters/generation.md](docs/parameters/generation.md) + evaluator doc |
| Load a config file | `-f path/to/file.yaml` | N/A (this is a file-loading action) | examples in [examples/](examples/) |
| Set output directory | `-o results_xxx` | `output.dir: results_xxx` (example in `run_suite.yaml`) | [examples/geneval/run_suite.yaml](examples/geneval/run_suite.yaml) |

### 2.3 Choose evaluator docs by `-e` key

| evaluator key (`-e`) | Documentation |
| --- | --- |
| `geneval` | [docs/parameters/evaluators/geneval.md](docs/parameters/evaluators/geneval.md) |
| `ella` | [docs/parameters/evaluators/ella.md](docs/parameters/evaluators/ella.md) |
| `genaibench` | [docs/parameters/evaluators/genaibench.md](docs/parameters/evaluators/genaibench.md) |
| `t2i_corebench` | [docs/parameters/evaluators/t2i_corebench.md](docs/parameters/evaluators/t2i_corebench.md) |

### 2.4 Precedence rules when values conflict

1. If the same key appears multiple times, the later value wins.
2. CLI overrides config file values. Example: `-g steps=10` overrides `generation.steps` in YAML.
3. `-G` only overrides keys explicitly provided for that evaluator. Unspecified keys still come from global `-g` / `generation`.

### 2.5 Debug-friendly knob: `num_samples`

`num_samples` is a simple way to test feasibility before committing to a full benchmark run.

- Purpose: run only a small subset of prompts/samples so you can verify that the full pipeline is healthy.
- Typical debug values: `num_samples=2`, `num_samples=4`, or `num_samples=8`.
- Why it helps: you can quickly catch config mistakes, missing dependencies, OOM issues, and evaluator errors.
- Recommended workflow: begin with a small `num_samples`, then scale up once outputs and metrics look sane.

Example:

```bash
uv run t2i-eval \
  -m diffusers \
  -a pretrained=runwayml/stable-diffusion-v1-5 \
  -g steps=20,seed=42 \
  -e geneval -E num_samples=4 \
  -o results_debug
```

---

## 3. CLI Quick Reference

This table is intentionally compact. When you need accepted values or defaults, follow the links in the last column.

| Flag | Purpose | Where to find details |
| --- | --- | --- |
| `-m, --model` | Select model registry key | [docs/parameters/models.md](docs/parameters/models.md) |
| `-a, --model-args` | Set model initialization args | [docs/parameters/models.md](docs/parameters/models.md) |
| `-g, --gen` | Set global generation args | [docs/parameters/generation.md](docs/parameters/generation.md) |
| `-e, --eval` | Add evaluator(s) | [docs/parameters/overview.md](docs/parameters/overview.md) |
| `-E, --eval-args` | Set evaluator args | [docs/parameters/evaluators/geneval.md](docs/parameters/evaluators/geneval.md), task specs for [Ella](docs/parameters/evaluators/ella.md) and [GenAI-Bench](docs/parameters/evaluators/genaibench.md) |
| `-G, --eval-gen` | Override generation args per evaluator | [docs/parameters/generation.md](docs/parameters/generation.md) + evaluator docs |
| `-f, --file` | Load YAML/JSON config | examples in [examples/](examples/) |
| `-o, --output-dir` | Set output directory | [examples/geneval/run_suite.yaml](examples/geneval/run_suite.yaml) |
| `--fail-fast` | Stop after first failure | Example E in this README |
| `--no-summary` | Suppress per-evaluator stdout summary | `t2i-eval --help` |
| `--quiet / --verbose` | Control log verbosity | `t2i-eval --help` |

---

## 4. Parameter-focused Troubleshooting

| Symptom | How to check |
| --- | --- |
| `Model 'xxx' is not registered` | Verify `-m` is a supported key in [docs/parameters/models.md](docs/parameters/models.md). |
| `Evaluator 'xxx' is not registered` | Verify `-e` spelling and confirm evaluator support in docs. |
| `extra fields not permitted` | You passed unsupported keys. Check exact key names in the corresponding parameter doc. |
| Unexpected results | Verify `seed`, `steps`, and `guidance_scale` were not overridden later by `-G` or another repeated flag. |

---

## 5. Source Entry Points

- CLI entrypoint: `src/t2i_eval/cli/main.py`
- Config merging: `src/t2i_eval/cli/config_loader.py`
- Runner orchestration: `src/t2i_eval/cli/runner.py`
- Registry: `src/t2i_eval/core/registry.py`
- Core schema: `src/t2i_eval/core/schema.py`

For day-to-day use, most readers only need Sections 0 through 2.
