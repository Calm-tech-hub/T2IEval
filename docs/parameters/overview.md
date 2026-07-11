# Parameter Reference Overview

This directory explains every CLI flag and the underlying configuration objects so you can edit configs confidently without reverse-engineering the source. Use the following guide:

| Document | What it covers |
| --- | --- |
| [`models.md`](models.md) | Registry keys (`-m`) and their supported `-a/--model-args` |
| [`generation.md`](generation.md) | Global generation parameters (`-g`) and how overrides (`-G`) merge |
| [`evaluators/geneval.md`](evaluators/geneval.md) | GenEval-specific evaluator args and generation overrides |
| [`evaluators/ella.md`](evaluators/ella.md) | ELLA task spec for the exam package |
| [`evaluators/genaibench.md`](evaluators/genaibench.md) | GenAI-Bench task spec for the exam package |
| [`evaluators/t2i_corebench.md`](evaluators/t2i_corebench.md) | T2I-CoReBench generation/existing-image and local-Qwen evaluation |

Recommended workflow:

1. Pick a model entry in `models.md` to learn which arguments are required (e.g., `pretrained`, `dtype`), plus sample commands.
2. Tune global sampling behavior using `generation.md` defaults/tips.
3. For each evaluator you plan to run, open its dedicated markdown to discover valid keys (`num_samples`, `split`, etc.) before writing `-E` or YAML overrides.

If you extend the framework with new models/evaluators, remember to update the relevant section here so future users know which parameters are accepted.
