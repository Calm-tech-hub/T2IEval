# Extending T2IEval

This guide shows how to add custom models and evaluators so you can benchmark proprietary pipelines.

## 1. Adding a model

1. Create a class that inherits `t2i_eval.core.model.BaseModel`. Implement at least:
   - `__init__(self, **kwargs)` to store init args.
   - `load(self)` (optional) for heavy assets.
   - `generate_batch(self, configs: list[GenerationConfig]) -> list[GenerationResult]`.
   - `unload(self)` to free GPU memory.
2. Register it in `src/t2i_eval/model/__init__.py`:
   ```python
   from .my_model import MyModel
   registry.register_model("my_model", MyModel)
   ```
3. Document its CLI arguments in `docs/parameters/models.md`.
4. Add a smoke test (e.g., `tests/my_model_test.py`) that calls `generate_batch` with a dummy config.

## 2. Adding an evaluator

1. Decide between:
   - `SimpleEvaluator`: compose loader → preprocessors → generator → postprocessors → aggregators.
   - Custom `BaseEvaluator`: implement `evaluate(model)` manually if flow is unusual.
2. Define a config dataclass (`SimpleEvalConfig` or `EvaluatorConfig` derivative) with explicit fields. Set `model_config = ConfigDict(extra="forbid")` to enforce parameter validation.
3. Register the evaluator in `src/t2i_eval/core/registry.py` using `@register_evaluator("my_eval")`.
4. Provide a paired test under `tests/` (policy: evaluator/test delivered together). Tests can mock `BaseModel` to avoid heavy generation.
5. Document CLI knobs in `docs/parameters/evaluators/my_eval.md`.

## 3. Wiring the CLI

Once models/evaluators are registered, `t2i-eval` automatically discovers them at startup (see `src/t2i_eval/cli/main.py` importing the registries). Users can invoke them with:

```bash
uv run t2i-eval \
  -m my_model -a foo=bar \
  -g steps=10 \
  -e my_eval -E custom_param=123
```

Remember to update README (supported evaluators section) and include example commands/configs so others know how to use your additions.
