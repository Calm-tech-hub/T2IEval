import json
import os
from pathlib import Path
from typing import Any

from t2i_eval.core.schema import GenerationResult


def _detect_process_info() -> tuple[int, int]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass

    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    if rank_env is not None and world_size_env is not None:
        try:
            return int(rank_env), int(world_size_env)
        except ValueError:
            pass

    return 0, 1


def save_results(
    eval_results: list[tuple[GenerationResult, dict[str, Any]]],
    sample_dir: str,
    process_index: int | None = None,
    num_processes: int | None = None,
    use_shared_index: bool = True,
) -> list[tuple[GenerationResult, dict[str, Any]]]:
    def default_serializer(obj: Any) -> Any:
        if hasattr(obj, "to_json"):
            return obj.to_json()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    if process_index is None or num_processes is None:
        detected_process_index, detected_num_processes = _detect_process_info()
        if process_index is None:
            process_index = detected_process_index
        if num_processes is None:
            num_processes = detected_num_processes

    output_dir = Path(sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, (generation_result, metadata) in enumerate(eval_results):
        if use_shared_index and num_processes > 1:
            # Keep ids globally unique without rank subdirectories.
            output_sample_idx = sample_idx * num_processes + process_index
        else:
            output_sample_idx = sample_idx

        sample_output_dir = output_dir / f"{output_sample_idx:06d}"
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        for image_idx, image in enumerate(generation_result.images):
            image_path = sample_output_dir / f"{image_idx:02d}.png"
            image.save(image_path)

        metadata_path = sample_output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                metadata, file, indent=2, ensure_ascii=False, default=default_serializer
            )

    return eval_results
