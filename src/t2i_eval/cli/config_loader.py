"""Helpers to merge CLI args and optional config files into a RunConfig.

Merge precedence: defaults -> config file -> CLI kwargs.
Evaluation order: CLI-declared ``-e`` names first (in the order given), then
any evaluations that exist only in the config file (preserving their order in
the file).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from . import schemas
from .utils import parse_scoped_kwargs


def _load_config_file(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text()
    if not text.strip():
        return {}

    if path.suffix.lower() in {".json"}:
        return json.loads(text)

    # Default to YAML
    return yaml.safe_load(text) or {}


def _build_evaluation_specs(evals_data: Dict[str, Dict] | List[Dict]) -> List[schemas.EvaluationSpec]:
    specs: List[schemas.EvaluationSpec] = []

    if isinstance(evals_data, dict):
        items: Iterable = evals_data.items()
    elif isinstance(evals_data, list):
        # fallback: list of {name:..., eval_args:..., gen_args:...}
        items = [(item.get("name"), item) for item in evals_data if isinstance(item, dict)]
    else:
        return specs

    for name, data in items:
        if not name:
            continue
        data = data or {}
        eval_args = data.get("eval_args", {}) if isinstance(data, dict) else {}
        gen_override = data.get("gen_args") or data.get("gen_override") or {}
        specs.append(
            schemas.EvaluationSpec(
                name=name,
                eval_args=dict(eval_args),
                gen_override=dict(gen_override),
            )
        )

    return specs


def _merge_model(config: schemas.RunConfig, name: str | None, args: Dict) -> None:
    if name:
        config.model.name = name
    if args:
        config.model.args.update(args)


def _merge_generation(config: schemas.RunConfig, params: Dict) -> None:
    if params:
        config.generation.params.update(params)


def _merge_output(config: schemas.RunConfig, output_dir: str | None) -> None:
    if output_dir:
        config.output.dir = output_dir


def _merge_evaluations(
    config: schemas.RunConfig,
    cli_eval_order: List[str],
    cli_eval_args: Dict[str, Dict],
    cli_eval_gen: Dict[str, Dict],
    config_evals: List[schemas.EvaluationSpec],
) -> None:
    config_lookup = {spec.name: spec for spec in config_evals}
    added = set()

    # CLI-declared evals first
    for name in cli_eval_order:
        base = config_lookup.get(name, schemas.EvaluationSpec(name=name))
        base.eval_args.update(cli_eval_args.get(name, {}))
        base.gen_override.update(cli_eval_gen.get(name, {}))
        config.evaluations.append(base)
        added.add(name)

    # Append any remaining config-only evals, preserving their original order
    for spec in config_evals:
        if spec.name in added:
            continue
        spec.eval_args.update(cli_eval_args.get(spec.name, {}))
        spec.gen_override.update(cli_eval_gen.get(spec.name, {}))
        config.evaluations.append(spec)


def _as_scoped_overrides(value) -> Dict[str, Dict]:
    """Accept either a scoped string (name:key=value,...) or a dict-of-dicts."""

    if not value:
        return {}
    if isinstance(value, str):
        return parse_scoped_kwargs(value)
    if isinstance(value, dict):
        # Assume dict-of-dicts; keep as-is but copy defensively.
        return {k: dict(v or {}) for k, v in value.items()}
    raise TypeError(f"Expected str or dict for scoped overrides, got {type(value)!r}")


def _merge_scoped_overrides(base: Dict[str, Dict], incoming: Dict[str, Dict]) -> Dict[str, Dict]:
    """Deep-merge two dict-of-dicts: base[scope].update(incoming[scope])."""

    if not incoming:
        return base
    for scope, kv in incoming.items():
        base.setdefault(scope, {}).update(kv or {})
    return base


def load_run_config(cli_kwargs: Dict) -> schemas.RunConfig:
    """Create a RunConfig from defaults, optional config file, then CLI kwargs."""

    cfg = schemas.RunConfig.default()
    cfg.fail_fast = bool(cli_kwargs.get("fail_fast", False))

    file_path = cli_kwargs.get("file")
    if file_path:
        data = _load_config_file(Path(file_path))

        _merge_model(cfg, data.get("model"), data.get("model_args", {}))
        _merge_generation(cfg, data.get("generation", {}))
        _merge_output(cfg, data.get("output", {}).get("dir") if isinstance(data.get("output"), dict) else data.get("output_dir"))

        cfg_evals = _build_evaluation_specs(data.get("evaluations", {}))
    else:
        cfg_evals = []

    # Apply CLI overrides
    _merge_model(cfg, cli_kwargs.get("model"), cli_kwargs.get("model_args", {}))
    _merge_generation(cfg, cli_kwargs.get("gen", {}))
    _merge_output(cfg, cli_kwargs.get("output_dir"))

    # CLI parity: accept either dicts (e.g. tests) or raw strings using `name:key=value`.
    # Prefer explicit dicts over raw strings when both are provided.
    cli_eval_args = _merge_scoped_overrides(
        _as_scoped_overrides(cli_kwargs.get("eval_args_raw")),
        _as_scoped_overrides(cli_kwargs.get("eval_args")),
    )
    cli_eval_gen = _merge_scoped_overrides(
        _as_scoped_overrides(cli_kwargs.get("eval_gen_raw")),
        _as_scoped_overrides(cli_kwargs.get("eval_gen")),
    )

    _merge_evaluations(
        cfg,
        cli_kwargs.get("eval", []) or [],
        cli_eval_args,
        cli_eval_gen,
        cfg_evals,
    )

    # If no evals were declared anywhere, leave the list empty (defaults already set)
    return cfg


__all__ = ["load_run_config"]
