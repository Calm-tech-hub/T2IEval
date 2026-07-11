"""Orchestrates model/evaluator lifecycle for `t2i-eval run`."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

from ..core import registry
from ..core.schema import GenerationConfig
from . import schemas
from .artifacts import ArtifactWriter
from .logger import CliLogger


def _safe_name(name: str) -> str:
    """Return a filesystem-safe slug."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "unknown")


class Runner:
    def __init__(
        self,
        config: schemas.RunConfig,
        logger: CliLogger | None = None,
        accelerator=None,
        show_summary: bool = True,
    ):
        self.config = config
        self.logger = logger or CliLogger()
        self.accelerator = accelerator
        self._model = None
        self.show_summary = show_summary

    def run(self) -> list[schemas.EvaluationResult]:
        model = None
        results: list[schemas.EvaluationResult] = []

        for eval_spec in self.config.evaluations:
            gen_cfg = self._build_generation_config(eval_spec)
            artifact_writer = ArtifactWriter(
                self.config.output.dir,
                self.config.model,
                eval_spec,
                gen_cfg.model_dump(),
            )
            if self.config.output.write_artifacts:
                artifact_writer.initialize()
            caught_exception = None
            try:
                if model is None:
                    model = self._init_model()
                result, evaluator = self._run_single(model, eval_spec, gen_cfg, artifact_writer)
                sample_records = getattr(evaluator, "sample_records", None)
                if self.config.output.write_artifacts:
                    if sample_records is not None:
                        artifact_writer.write_samples(sample_records)
                    artifact_records = getattr(evaluator, "artifact_records", {})
                    for filename, records in artifact_records.items():
                        artifact_writer.write_jsonl(filename, records)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"[{eval_spec.name}] failed: {exc}")
                if self.show_summary:
                    self.logger.error(f"[{eval_spec.name}] FAILED: {exc}")
                caught_exception = exc
                result = schemas.EvaluationResult(
                    eval_name=eval_spec.name,
                    metrics={},
                    error="".join(traceback.format_exception(exc)),
                )
            output_path = self._write_result(result, gen_cfg)
            if self.config.output.write_artifacts:
                artifact_writer.write_result(result)
            if self.show_summary and result.error is None:
                summary = self._format_metrics(result.metrics)
                self.logger.info(f"[{eval_spec.name}] metrics={summary} -> {output_path}")
            results.append(result)
            if caught_exception is not None and self.config.fail_fast:
                raise caught_exception
        return results

    # --- internals ----------------------------------------------------- #

    def _init_model(self):
        if self._model is not None:
            return self._model

        model_cls = registry.get_model_class(self.config.model.name)
        if model_cls is None:
            raise ValueError(f"Model '{self.config.model.name}' is not registered")

        model = model_cls(**self.config.model.args)

        # Optional accelerator hook
        if self.accelerator and hasattr(model, "enable_accelerator"):
            try:
                model.enable_accelerator(self.accelerator)
            except Exception:  # noqa: BLE001
                # Best-effort; don't block run on optional feature
                self.logger.warn("enable_accelerator failed; continuing without it")

        if hasattr(model, "load"):
            model.load()

        self._model = model
        return model

    def _build_generation_config(self, eval_spec: schemas.EvaluationSpec) -> GenerationConfig:
        merged: dict = {}
        merged.update(self.config.generation.params or {})
        merged.update(eval_spec.gen_override or {})
        return GenerationConfig(**merged)

    def _run_single(
        self,
        model,
        eval_spec: schemas.EvaluationSpec,
        gen_cfg: GenerationConfig,
        artifact_writer: ArtifactWriter,
    ) -> tuple[schemas.EvaluationResult, object]:
        evaluator_cls = registry.get_evaluator_class(eval_spec.name)
        if evaluator_cls is None:
            raise ValueError(f"Evaluator '{eval_spec.name}' is not registered")

        eval_kwargs = dict(eval_spec.eval_args or {})
        eval_kwargs.setdefault("device", self.config.model.args.get("device", "cuda"))
        if self.accelerator is not None:
            eval_kwargs.setdefault("accelerator", self.accelerator)
        if self.config.output.save_images or self.config.output.resume:
            eval_kwargs.setdefault("sample_dir", str(artifact_writer.images_dir))
        else:
            eval_kwargs.setdefault("sample_dir", None)
        eval_kwargs.setdefault("resume", self.config.output.resume)
        eval_kwargs.setdefault("generation_config", gen_cfg.model_dump())

        try:
            evaluator = evaluator_cls(**eval_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to initialize evaluator '{eval_spec.name}': {exc}"
            ) from exc
        data = evaluator.evaluate(model) or {}

        result = schemas.EvaluationResult(
            eval_name=eval_spec.name,
            metrics=data,
            error=None,
        )

        return result, evaluator

    def _write_result(self, result: schemas.EvaluationResult, gen_cfg: GenerationConfig) -> Path:
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"results_{_safe_name(result.eval_name)}_{_safe_name(self.config.model.name)}.json"
        path = output_dir / filename

        payload = {
            "eval_name": result.eval_name,
            "model_name": self.config.model.name,
            "model_args": self.config.model.args,
            "generation_config": gen_cfg.model_dump(),
            "metrics": result.metrics,
            "error": result.error,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _format_metrics(self, metrics: dict) -> str:
        try:
            return json.dumps(metrics, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            # Fallback if metrics contain non-serializable objects.
            return str(metrics)


__all__ = ["Runner"]
