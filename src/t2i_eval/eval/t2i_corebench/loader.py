import copy
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ...core.benchmark import BenchmarkSample
from ...core.schema import GenerationConfig

COMPOSITION_DIMENSIONS = ("C-MI", "C-MA", "C-MR", "C-TR")
REASONING_DIMENSIONS = (
    "R-LR",
    "R-BR",
    "R-HR",
    "R-PR",
    "R-GR",
    "R-AR",
    "R-CR",
    "R-RR",
)
ALL_DIMENSIONS = COMPOSITION_DIMENSIONS + REASONING_DIMENSIONS


def normalize_dimensions(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return list(ALL_DIMENSIONS)
        dimensions = [part.strip().upper() for part in value.split(",")]
    else:
        dimensions = [str(part).strip().upper() for part in value]

    dimensions = [dimension for dimension in dimensions if dimension]
    unknown = sorted(set(dimensions) - set(ALL_DIMENSIONS))
    if unknown:
        raise ValueError(
            f"Unknown T2I-CoReBench dimensions {unknown}; "
            f"expected a subset of {list(ALL_DIMENSIONS)}."
        )
    if not dimensions:
        raise ValueError("At least one T2I-CoReBench dimension is required.")
    return list(dict.fromkeys(dimensions))


def resolve_data_dir(
    data_dir: str | None,
    dataset_name: str,
    cache_dir: str | None = None,
) -> Path:
    if data_dir is not None:
        path = Path(data_dir).expanduser().resolve()
    else:
        snapshot = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                cache_dir=cache_dir,
                allow_patterns=["*.json", "data/*.json"],
            )
        )
        path = snapshot / "data" if (snapshot / "data").is_dir() else snapshot

    if not path.is_dir():
        raise FileNotFoundError(f"T2I-CoReBench data directory not found: {path}")
    return path


def _image_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-(\d+)$", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def _find_images(
    image_dir: Path,
    dimension: str,
    prompt_id: str,
    extensions: tuple[str, ...],
) -> list[Path]:
    dimension_dir = image_dir / dimension
    search_dir = dimension_dir if dimension_dir.is_dir() else image_dir
    matches = []
    for extension in extensions:
        suffix = extension if extension.startswith(".") else f".{extension}"
        matches.extend(search_dir.glob(f"{prompt_id}-*{suffix}"))
        matches.extend(search_dir.glob(f"{prompt_id}-*{suffix.upper()}"))
    return sorted(set(path.resolve() for path in matches), key=_image_sort_key)


def _load_dimension(path: Path, dimension: str) -> dict[str, dict[str, Any]]:
    file_path = path / f"{dimension}.json"
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing T2I-CoReBench dimension file: {file_path}"
        )
    with file_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, dict):
        raise ValueError(f"Expected an object in {file_path}, got {type(records)!r}")
    return records


def load_corebench_records(
    *,
    data_dir: str | None,
    dataset_name: str,
    cache_dir: str | None,
    dimensions: str | list[str] | tuple[str, ...],
    default_config: GenerationConfig,
    num_prompts: int | None,
    image_dir: str | None,
    strict_images: bool,
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> list[BenchmarkSample]:
    """Load official checklist JSON and expand every prompt into image samples."""

    if num_prompts is not None and num_prompts <= 0:
        raise ValueError("num_prompts must be positive or null.")
    images_per_prompt = default_config.num_images_per_prompt
    if images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive.")

    benchmark_dir = resolve_data_dir(data_dir, dataset_name, cache_dir)
    selected_dimensions = normalize_dimensions(dimensions)
    existing_root = Path(image_dir).expanduser().resolve() if image_dir else None
    if existing_root is not None and not existing_root.is_dir():
        raise FileNotFoundError(
            f"T2I-CoReBench existing-image directory not found: {existing_root}"
        )

    samples: list[BenchmarkSample] = []
    generated_index = 0
    for dimension in selected_dimensions:
        records = _load_dimension(benchmark_dir, dimension)
        items = list(records.items())
        if num_prompts is not None:
            items = items[:num_prompts]

        for prompt_index, (prompt_id, record) in enumerate(items):
            prompt = str(record.get("Prompt", "")).strip()
            checklist = record.get("Checklist")
            if not prompt or not isinstance(checklist, list) or not checklist:
                raise ValueError(
                    f"Invalid T2I-CoReBench record {prompt_id!r} in {dimension}."
                )

            paths: list[Path | None]
            if existing_root is None:
                paths = [None] * images_per_prompt
            else:
                found = _find_images(
                    existing_root,
                    dimension,
                    str(prompt_id),
                    image_extensions,
                )
                if strict_images and len(found) < images_per_prompt:
                    raise FileNotFoundError(
                        f"Expected {images_per_prompt} images for {prompt_id}, "
                        f"found {len(found)} below {existing_root}."
                    )
                paths = found[:images_per_prompt]
                if not paths:
                    continue

            for image_index, image_path in enumerate(paths):
                generation_config = default_config.model_copy(deep=True)
                generation_config.prompt = prompt
                generation_config.num_images_per_prompt = 1
                if default_config.seed is not None:
                    generation_config.seed = default_config.seed + generated_index
                if image_path is not None:
                    generation_config.extra_kwargs["image_path"] = str(image_path)

                sample_id = f"{prompt_id}-{image_index}"
                metadata = {
                    "prompt_id": str(prompt_id),
                    "sample_id": sample_id,
                    "image_index": image_index,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "dimension": dimension,
                    "main_class": str(record.get("Main Class", "")),
                    "sub_class": str(record.get("Sub Class", "")),
                    "checklist": copy.deepcopy(checklist),
                    "remark": record.get("Remark"),
                }
                if image_path is not None:
                    metadata["source_image_path"] = str(image_path)

                samples.append(
                    BenchmarkSample(
                        sample_id=sample_id,
                        prompt=prompt,
                        generation_config=generation_config,
                        metadata=metadata,
                    )
                )
                generated_index += 1

    if not samples:
        source = f" below {existing_root}" if existing_root else ""
        raise ValueError(f"No T2I-CoReBench samples were loaded{source}.")
    return samples


__all__ = [
    "ALL_DIMENSIONS",
    "COMPOSITION_DIMENSIONS",
    "REASONING_DIMENSIONS",
    "load_corebench_records",
    "normalize_dimensions",
]
