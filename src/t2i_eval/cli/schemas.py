"""Lightweight CLI config schemas used by the unified `t2i-eval run` command.

These dataclasses intentionally keep defaults minimal; richer validation and
merge logic will be layered in later tasks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    """Model selection plus initialization kwargs."""

    name: str = "diffusers"
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationSpec:
    """Generation-time overrides; contents are model-dependent."""

    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSpec:
    """One evaluator invocation with optional generation overrides."""

    name: str
    eval_args: Dict[str, Any] = field(default_factory=dict)
    gen_override: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputSpec:
    """Output directory information."""

    dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Structured result emitted by Runner for each evaluation."""

    eval_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    """Top-level configuration passed to the Runner."""

    model: ModelSpec
    generation: GenerationSpec
    evaluations: List[EvaluationSpec] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    fail_fast: bool = False

    @classmethod
    def default(cls) -> "RunConfig":
        """Return a RunConfig populated with sensible defaults."""

        return cls(
            model=ModelSpec(),
            generation=GenerationSpec(),
            output=OutputSpec(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Construct a RunConfig from a dict; minimal parsing for now."""

        model_data = data.get("model", {})
        gen_data = data.get("generation", {})
        eval_data = data.get("evaluations", [])
        output_data = data.get("output", {})

        model = ModelSpec(**model_data) if isinstance(model_data, dict) else ModelSpec()
        generation = (
            GenerationSpec(**gen_data) if isinstance(gen_data, dict) else GenerationSpec()
        )
        evaluations = [EvaluationSpec(**item) for item in eval_data if isinstance(item, dict)]
        output = OutputSpec(**output_data) if isinstance(output_data, dict) else OutputSpec()

        fail_fast = bool(data.get("fail_fast", False))
        return cls(
            model=model,
            generation=generation,
            evaluations=evaluations,
            output=output,
            fail_fast=fail_fast,
        )

    def merge_dict(self, override: Dict[str, Any]) -> "RunConfig":
        """Return a new RunConfig with a shallow dict merge applied.

        Full merge semantics will be implemented later; this keeps the API stable.
        """

        base = self.to_dict()
        base.update(override or {})
        return RunConfig.from_dict(base)
