from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

TEMPLATE_VERSION = "official-2026-01"


@dataclass(frozen=True)
class JudgeRequest:
    request_id: str
    image_path: str
    prompt: str
    question: str


@dataclass
class JudgeResponse:
    request_id: str
    answer: str | None
    score: int | None
    raw_response: str
    error: str | None = None
    cached: bool = False


class CoreBenchJudge(Protocol):
    model_name: str

    def score(self, requests: list[JudgeRequest]) -> list[JudgeResponse]: ...

    def close(self) -> None: ...


def parse_binary_answer(text: str) -> tuple[str | None, int | None]:
    final_text = text.split("</think>")[-1].strip().lower()
    if re.search(r"\byes\b", final_text):
        return "yes", 1
    if re.search(r"\bno\b", final_text):
        return "no", 0
    return None, None


class SubprocessQwenJudge:
    """Run the official-style Qwen/vLLM evaluator in an isolated Python env."""

    def __init__(
        self,
        *,
        model_name: str,
        python_executable: str | None = None,
        cache_path: str | None = None,
        batch_size: int = 64,
        max_rounds: int = 3,
        initial_max_tokens: int = 512,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        request_chunk_size: int = 4096,
        timeout: float | None = None,
    ):
        self.model_name = model_name
        self.python_executable = python_executable or sys.executable
        self.cache_path = Path(cache_path).expanduser().resolve() if cache_path else None
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.initial_max_tokens = initial_max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.request_chunk_size = request_chunk_size
        self.timeout = timeout

        if batch_size <= 0 or max_rounds <= 0 or initial_max_tokens <= 0:
            raise ValueError("Qwen judge batch and retry settings must be positive.")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1.")
        if tensor_parallel_size <= 0 or request_chunk_size <= 0:
            raise ValueError("Qwen judge parallel and chunk sizes must be positive.")

        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, JudgeResponse]:
        cache: dict[str, JudgeResponse] = {}
        if self.cache_path is None or not self.cache_path.is_file():
            return cache
        with self.cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                    key = payload.pop("cache_key")
                    cache[key] = JudgeResponse(**payload)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return cache

    def _cache_key(self, request: JudgeRequest) -> str:
        digest = hashlib.sha256()
        with Path(request.image_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(request.prompt.encode("utf-8"))
        digest.update(request.question.encode("utf-8"))
        digest.update(self.model_name.encode("utf-8"))
        digest.update(TEMPLATE_VERSION.encode("ascii"))
        return digest.hexdigest()

    def _append_cache(self, entries: list[tuple[str, JudgeResponse]]) -> None:
        if self.cache_path is None or not entries:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            for key, response in entries:
                payload = {"cache_key": key, **asdict(response)}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _run_worker(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        worker = Path(__file__).with_name("qwen_worker.py")
        with tempfile.TemporaryDirectory(prefix="t2i-corebench-qwen-") as directory:
            root = Path(directory)
            input_path = root / "requests.jsonl"
            output_path = root / "responses.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                for request in requests:
                    handle.write(json.dumps(asdict(request), ensure_ascii=False) + "\n")

            command = [
                self.python_executable,
                str(worker),
                "--model",
                self.model_name,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--batch-size",
                str(self.batch_size),
                "--max-rounds",
                str(self.max_rounds),
                "--initial-max-tokens",
                str(self.initial_max_tokens),
                "--gpu-memory-utilization",
                str(self.gpu_memory_utilization),
                "--tensor-parallel-size",
                str(self.tensor_parallel_size),
                "--request-chunk-size",
                str(self.request_chunk_size),
            ]
            completed = subprocess.run(
                command,
                check=False,
                stdout=None,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or "").strip()
                raise RuntimeError(
                    "Local Qwen judge worker failed. Ensure the configured "
                    "Python environment contains vLLM, Transformers with "
                    f"Qwen3.5 support, and qwen-vl-utils.\n{detail}"
                )
            if not output_path.is_file():
                raise RuntimeError("Local Qwen judge worker produced no result file.")

            responses = []
            with output_path.open(encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    raw = str(payload.get("raw_response", ""))
                    answer, score = parse_binary_answer(raw)
                    responses.append(
                        JudgeResponse(
                            request_id=str(payload["request_id"]),
                            answer=answer,
                            score=score,
                            raw_response=raw,
                            error=payload.get("error")
                            or (None if score is not None else "invalid_binary_answer"),
                        )
                    )
            return responses

    def score(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        if not requests:
            return []

        keys = {request.request_id: self._cache_key(request) for request in requests}
        results: dict[str, JudgeResponse] = {}
        missing = []
        for request in requests:
            cached = self._cache.get(keys[request.request_id])
            if cached is None:
                missing.append(request)
            else:
                results[request.request_id] = JudgeResponse(
                    **{**asdict(cached), "request_id": request.request_id, "cached": True}
                )

        new_responses = self._run_worker(missing) if missing else []
        by_request = {response.request_id: response for response in new_responses}
        cache_entries = []
        for request in missing:
            response = by_request.get(request.request_id)
            if response is None:
                response = JudgeResponse(
                    request_id=request.request_id,
                    answer=None,
                    score=None,
                    raw_response="",
                    error="missing_worker_response",
                )
            results[request.request_id] = response
            if response.score in (0, 1):
                key = keys[request.request_id]
                self._cache[key] = response
                cache_entries.append((key, response))
        self._append_cache(cache_entries)

        return [results[request.request_id] for request in requests]

    def close(self) -> None:
        return None


__all__ = [
    "CoreBenchJudge",
    "JudgeRequest",
    "JudgeResponse",
    "SubprocessQwenJudge",
    "parse_binary_answer",
]
