"""Typed contracts shared by benchmark loaders and evaluation stages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .schema import GenerationConfig, GenerationResult


class BenchmarkSample(BaseModel):
    """Normalized sample emitted by a benchmark loader."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    prompt: str
    generation_config: GenerationConfig
    metadata: dict[str, Any] = Field(default_factory=dict)
    references: dict[str, Any] = Field(default_factory=dict)


class SampleEvaluation(BaseModel):
    """Normalized sample-level evaluation artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    prompt: str = ""
    generation: GenerationResult
    image_paths: list[str] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class BenchmarkLoader(Protocol):
    def load(self) -> Iterable[BenchmarkSample]: ...


@runtime_checkable
class BenchmarkPostprocessor(Protocol):
    def process(
        self,
        sample: BenchmarkSample,
        generation: GenerationResult,
    ) -> SampleEvaluation: ...


@runtime_checkable
class BenchmarkAggregator(Protocol):
    def aggregate(self, results: Iterable[SampleEvaluation]) -> dict[str, Any]: ...
