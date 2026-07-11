from functools import partial
from typing import Any

import clip_benchmark.metrics.zeroshot_classification as zsc
import numpy as np
import open_clip
import torch
from pydantic import Field

from ...core.registry import register_evaluator
from ...core.schema import GenerationConfig, GenerationResult
from .. import aggregator, processor
from ..aggregator import Reduction
from ..loader import load_hf_records
from ..simple_evaluator import SimpleEvalConfig, SimpleEvaluator
from .utils import COLORS, color_classification, compute_iou, relative_position

# Helper to suppress zsc tqdm
zsc.tqdm = lambda it, *args, **kwargs: it

CATEGORIES = [
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
]


class GenevalConfig(SimpleEvalConfig):
    """Configuration for Geneval evaluator."""

    # HuggingFace grouped dataset (uploaded as parquet/jsonl-backed table)
    dataset_name: str = "Vertsineu/geneval"
    split: str = "validation"

    # options: all / single_object / two_object / counting / colors / position / color_attr
    category: str = "all"

    # official Geneval scripts use a fixed seed by default
    seed: int = 42

    # total number of prompts to sample after optional category filtering
    num_samples: int | None = None

    # Local/remote detector model source for Geneval object detection.
    detector_model_id: str = "facebook/mask2former-swin-small-coco-instance"
    detector_local_files_only: bool = False

    # Generation settings are managed by a dedicated GenerationConfig subclass.
    generation_config: "GenevalGenerationConfig" = Field(
        default_factory=lambda: GenevalGenerationConfig()
    )


class GenevalGenerationConfig(GenerationConfig):
    """Default generation config template for GenEval runs."""

    steps: int = 50
    guidance_scale: float = 9.0
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    num_images_per_prompt: int = 4


# ==================== Helper Functions ====================


def _geneval_row_to_eval_items(
    row: dict[str, Any],
    config: GenerationConfig,
) -> list[tuple[GenerationConfig, dict[str, Any]]]:
    category = row["category"]
    prompt = row["prompt"]
    metadata = row["metadata"]
    config.prompt = prompt
    metadata = {
        "category": category,
        "prompt": prompt,
        "metadata": metadata,
    }

    return [(config, metadata)]


class GenevalPostprocessor:
    """Postprocessor for evaluating individual images."""

    def __init__(
        self,
        device: str | torch.device,
        detector_model_id: str,
        detector_local_files_only: bool = False,
    ):
        self.device = device
        self.detector_model_id = detector_model_id
        self.detector_local_files_only = detector_local_files_only

        # Load models
        self.object_detector = None
        self.processor = None
        self.clip_model = None
        self.clip_transform = None
        self.clip_tokenizer = None

        # Params
        self.threshold = 0.3
        self.counting_threshold = 0.9
        self.max_objects = 16
        self.nms_threshold = 1.0
        self.position_threshold = 0.1
        self.mask_threshold = 0.5
        # Map detector label -> Geneval/COCO label (Geneval metadata uses COCO-style names).
        # Some detectors/configs expose VOC-style labels (e.g., "aeroplane", "tvmonitor", "sofa").
        self.detector_label_aliases: dict[str, str] = {
            # Common HF COCO-vs-Geneval mismatches
            "mouse": "computer mouse",
            "remote": "tv remote",
            "keyboard": "computer keyboard",
            # VOC-style names sometimes appear in configs
            "aeroplane": "airplane",
            "motorbike": "motorcycle",
            "sofa": "couch",
            "tvmonitor": "tv",
            # No-space variants
            "diningtable": "dining table",
            "pottedplant": "potted plant",
        }
        # Normalized (no punctuation/space) aliases for robustness across label spellings.
        self._detector_label_aliases_norm: dict[str, str] = {
            self._norm_label(k): v for k, v in self.detector_label_aliases.items()
        }

    @staticmethod
    def _norm_label(label: str) -> str:
        # Keep it simple and deterministic: lowercase + strip non-alnum.
        import re

        s = label.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    def _to_geneval_label(self, raw_label: str) -> str:
        # Exact first (fast path)
        mapped = self.detector_label_aliases.get(raw_label)
        if mapped is not None:
            return mapped
        # Normalized fallback
        mapped = self._detector_label_aliases_norm.get(self._norm_label(raw_label))
        if mapped is not None:
            return mapped
        return raw_label

    def _load_models(self):
        if self.object_detector is not None:
            return

        print("Loading Geneval models (Mask2Former, CLIP)...")

        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        # Object detector: supports both local directory and HF repo id.
        model_id = self.detector_model_id
        try:
            self.processor = Mask2FormerImageProcessor.from_pretrained(
                model_id,
                local_files_only=self.detector_local_files_only,
            )
            self.object_detector = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                local_files_only=self.detector_local_files_only,
            ).to(self.device)  # type: ignore
            self.object_detector.eval()
        except Exception as e:
            print(f"Error initializing detector: {e}")
            raise

        # CLIP
        clip_arch = "ViT-L-14"
        self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained="openai", device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_arch)

        print("Geneval models loaded.")

    def _unload_models(self):
        self.object_detector = None
        self.processor = None
        self.clip_model = None
        self.clip_transform = None
        self.clip_tokenizer = None

    def _evaluate_check_logic(self, image, objects, metadata):
        correct = True
        reason = []
        matched_groups = []

        # Check for expected objects
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[: req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}>={req['count']}, found {len(found_objects)}"
                )
            else:
                if "color" in req:
                    # Color check
                    colors = color_classification(
                        image,
                        found_objects,
                        classname,
                        self.clip_model,
                        self.clip_tokenizer,
                        self.device,
                        self.clip_transform,
                    )
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    # Relative position check
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(
                                    obj, target_obj, self.position_threshold
                                )
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)

        # Check for non-expected objects
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, found {len(objects[classname])}"
                )

        return correct, "\n".join(reason)

    def _evaluate_single_image(self, image, data):
        # HF Mask2Former inference
        assert (
            self.processor is not None and self.object_detector is not None
        ), "Models not loaded."
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.object_detector(**inputs)

        # Post-process for instance segmentation
        target_sizes = [(image.height, image.width)]
        confidence_threshold = (
            self.threshold
            if data["category"] != "counting"
            else self.counting_threshold
        )
        # Keep HF post-processing thresholding explicit for reproducibility.
        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold,
            mask_threshold=self.mask_threshold,
            return_binary_maps=True,
        )[0]

        segments_info = results["segments_info"]
        binary_maps = results["segmentation"]
        binary_maps_np = None if binary_maps is None else binary_maps.cpu().numpy()

        detected = {}

        id2label = self.object_detector.config.id2label

        for idx, segment in enumerate(segments_info):
            score = segment["score"]
            label_id = segment["label_id"]
            label_name = self._to_geneval_label(id2label[label_id])  # type: ignore

            # With return_binary_maps=True, segmentation has shape [num_instances, H, W].
            if binary_maps_np is None or idx >= len(binary_maps_np):
                continue
            mask = (binary_maps_np[idx] > 0).astype(np.uint8) * 255

            # Compute bbox from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue

            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox = np.array([x1, y1, x2, y2, score])

            if label_name not in detected:
                detected[label_name] = []

            detected[label_name].append((bbox, mask))

        final_detected = {}
        for classname in detected:
            candidates = detected[classname]
            # Sort by score descending
            candidates.sort(key=lambda x: x[0][4], reverse=True)

            candidates = candidates[: self.max_objects]

            kept = []
            while candidates:
                max_obj = candidates.pop(0)
                kept.append(max_obj)

                if self.nms_threshold < 1.0:
                    candidates = [
                        c
                        for c in candidates
                        if compute_iou(max_obj[0], c[0]) < self.nms_threshold
                    ]

            final_detected[classname] = kept

        # Pass the parsed metadata dict to _evaluate_check_logic
        is_correct, reason = self._evaluate_check_logic(
            image, final_detected, data["metadata"]
        )

        return {
            "correct": is_correct,
            "reason": reason,
        }

    def __call__(
        self, eval_results: list[tuple[GenerationResult, dict[str, Any]]]
    ) -> list[tuple[GenerationResult, dict[str, Any]]]:
        """Evaluate each generated image."""
        self._load_models()

        result = []
        for gen_result, metadata in eval_results:
            per_image_correct = []
            per_image_reason = []

            for image in gen_result.images:
                eval_result = self._evaluate_single_image(image, metadata)
                per_image_correct.append(bool(eval_result["correct"]))
                per_image_reason.append(eval_result["reason"])

            num_images = len(per_image_correct)
            metadata["per_image_correct"] = per_image_correct
            metadata["per_image_reason"] = per_image_reason
            metadata["num_images"] = num_images
            metadata["num_correct_images"] = int(sum(per_image_correct))
            metadata["correct"] = (
                float(sum(per_image_correct) / num_images) if num_images > 0 else 0.0
            )
            metadata["prompt_correct"] = bool(any(per_image_correct))
            metadata["reason"] = (
                per_image_reason[0] if len(per_image_reason) == 1 else per_image_reason
            )

            result.append((gen_result, metadata))

        self._unload_models()
        return result


# ==================== Main Evaluator ====================


@register_evaluator("geneval")
class GenevalEvaluator(SimpleEvaluator):
    """
    Evaluator implementation for Geneval.

    Reference: https://github.com/djghosh13/geneval
    """

    def __init__(self, **kwargs):
        config = GenevalConfig(**kwargs)
        if config.category != "all" and config.category not in CATEGORIES:
            raise ValueError(
                f"Unsupported Geneval category '{config.category}'. Supported: all, {', '.join(CATEGORIES)}"
            )

        config.loader = partial(
            load_hf_records,
            dataset_name=config.dataset_name,
            default_config=config.generation_config,
            row_to_eval_items=_geneval_row_to_eval_items,
            split=config.split,
            seed=config.seed,
            deserialize_columns=["metadata", "include", "exclude"],
        )

        config.preprocessor = []
        if config.category != "all":
            config.preprocessor.append(
                partial(processor.filter, conditions={"category": config.category})
            )
        if config.num_samples is not None:
            config.preprocessor.append(
                partial(
                    processor.sample,
                    seed=config.seed,
                    sample_size=config.num_samples,
                )
            )

        if config.accelerator is not None:
            device = config.accelerator.device
        else:
            device = config.device
        config.postprocessor = [
            GenevalPostprocessor(
                device=device,
                detector_model_id=config.detector_model_id,
                detector_local_files_only=config.detector_local_files_only,
            )
        ]

        config.aggregator = [
            partial(
                aggregator.aggregate_metric,
                value_key="correct",
                output_key="task_scores",
                reduction=Reduction.MEAN,
                group_by="category",
            ),
            partial(
                aggregator.aggregate_metric,
                value_key="correct",
                output_key="score",
                reduction=Reduction.MEAN,
                group_by="category",
                final_reduction=Reduction.MEAN,
            ),
        ]

        super().__init__(config)
