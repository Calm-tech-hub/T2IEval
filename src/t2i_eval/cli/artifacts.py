"""Structured, backwards-compatible run artifact persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import schemas


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _json_bytes(value: Any, *, indent: int | None = 2) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def _safe_name(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown").strip("_")


class ArtifactWriter:
    """Write one evaluator run without changing the legacy result contract."""

    def __init__(
        self,
        output_dir: str | Path,
        model: schemas.ModelSpec,
        evaluation: schemas.EvaluationSpec,
        generation_config: dict[str, Any],
    ):
        self.output_dir = Path(output_dir)
        self.model = model
        self.evaluation = evaluation
        self.generation_config = generation_config
        self.started_at = datetime.now(UTC)

        model_identity = str(model.args.get("pretrained") or model.name).rstrip("/")
        model_slug = _safe_name(model_identity.rsplit("/", 1)[-1])
        fingerprint_payload = {
            "model": model.to_dict(),
            "evaluation": evaluation.to_dict(),
            "generation_config": generation_config,
        }
        fingerprint = hashlib.sha256(_json_bytes(fingerprint_payload, indent=None)).hexdigest()[:12]
        self.run_id = f"{_safe_name(evaluation.name)}__{model_slug}__{fingerprint}"
        self.run_dir = self.output_dir / "runs" / self.run_id
        self.images_dir = self.run_dir / "images"

    def initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # A repeated run with the same deterministic config reuses the same
        # directory.  Remove only run-level terminal markers; sample images
        # remain available for explicit resume mode.
        (self.run_dir / "complete.marker").unlink(missing_ok=True)
        (self.run_dir / "errors.jsonl").unlink(missing_ok=True)
        config_payload = {
            "run_id": self.run_id,
            "model": self.model.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "generation_config": self.generation_config,
            "started_at": self.started_at.isoformat(),
        }
        _atomic_write(self.run_dir / "config.json", _json_bytes(config_payload))
        _atomic_write(self.run_dir / "environment.json", _json_bytes(self._environment()))

    def write_result(self, result: schemas.EvaluationResult) -> None:
        payload = {
            "eval_name": result.eval_name,
            "metrics": result.metrics,
            "error": result.error,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write(self.run_dir / "metrics.json", _json_bytes(payload))
        if result.error:
            (self.run_dir / "complete.marker").unlink(missing_ok=True)
            _atomic_write(
                self.run_dir / "errors.jsonl",
                _json_bytes({"scope": "run", "error": result.error}, indent=None) + b"\n",
            )
        else:
            (self.run_dir / "errors.jsonl").unlink(missing_ok=True)
            _atomic_write(self.run_dir / "complete.marker", b"complete\n")

    def write_samples(self, records: Iterable[dict[str, Any]]) -> None:
        self.write_jsonl("samples.jsonl", records)

    def write_jsonl(
        self,
        filename: str,
        records: Iterable[dict[str, Any]],
    ) -> None:
        path = Path(filename)
        if path.name != filename or path.suffix != ".jsonl":
            raise ValueError("Artifact JSONL filename must be a plain .jsonl name.")
        lines = [_json_bytes(record, indent=None) for record in records]
        data = b"\n".join(lines) + (b"\n" if lines else b"")
        _atomic_write(self.run_dir / filename, data)

    def _environment(self) -> dict[str, Any]:
        packages = {}
        for package in ("torch", "diffusers", "transformers", "datasets", "t2v-metrics"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None

        git_commit = None
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path.cwd(),
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if completed.returncode == 0:
                git_commit = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

        gpu = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu = {
                    "name": torch.cuda.get_device_name(0),
                    "count": torch.cuda.device_count(),
                    "cuda": torch.version.cuda,
                }
        except Exception:
            pass

        return {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "gpu": gpu,
            "git_commit": git_commit,
        }


__all__ = ["ArtifactWriter"]
