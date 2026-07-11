"""Isolated vLLM worker used by the framework-native T2I-CoReBench evaluator.

The system prompt and inference behavior are adapted from the official
Apache-2.0 T2I-CoReBench ``evaluate.py`` at revision
ebf7b7a0ac0da088f4664c50012b6b483ac6f02b. See THIRD_PARTY_NOTICES.md.
"""

import argparse
import json
import os
from pathlib import Path

SYSTEM_PROMPT = """
You are an AI quality auditor for text-to-image generation.

Your task is to analyze the given image and answer a yes/no question based solely on its visual content. The question may relate to the presence of a specific object, its attributes, or relationships between multiple elements in the image.

You will also be given the original prompt used to generate the image. The prompt may provide additional context to help interpret the question, but it must never be used to supply or assume visual details.
Your judgment must rely entirely on the image itself. The image must contain clear, unmistakable visual evidence to justify a "yes" answer — the prompt cannot compensate for missing or ambiguous content.

Respond with:
- "yes" only if the answer is **clearly and unambiguously** yes based solely on the visual content. The visual evidence must be **strong, definitive, and require no assumptions or guesses**.
- "no" in **all other cases** — including if the relevant visual detail is missing, unclear, ambiguous, partially shown, obscured, or only suggested.

Even if the image closely matches what is described in the prompt, you must rely on **visible evidence** alone. If the relevant detail cannot be confirmed visually with certainty, answer "no".
**Ambiguity equals no.**

For conditional questions, answer "yes" only if **both** the condition and the main clause are **clearly and unambiguously true** in the image. If **either part** is false or uncertain, respond "no".

Do **not** provide any explanation, justification, or extra text.
Only return a single word: either "yes" or "no".

Example input:
Prompt: "a golden retriever running in a grassy field under the sun"
Question: "Is there a sun in the image?"
Example output:
"yes"

Example input:
Prompt: "a white cat sitting on a red couch in a modern living room"
Question: "Is the couch is present, is it red in color?"
Example output:
"no"
""".strip()


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--initial-max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--request-chunk-size", type=int, default=4096)
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _extract_text(text: str, is_last_round: bool) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip() if is_last_round else ""


def main() -> None:
    args = _arguments()
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "qwen_worker.py requires vllm, a Qwen3.5-compatible transformers "
            "version, qwen-vl-utils, and torch."
        ) from exc

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm = LLM(
        model=args.model,
        max_model_len=25600,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_encoder_tp_mode="data",
        dtype="bfloat16",
        distributed_executor_backend="mp",
        reasoning_parser="qwen3",
        mm_processor_cache_type="shm",
        enable_prefix_caching=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    requests = _read_jsonl(args.input)
    responses: dict[str, dict] = {}

    for chunk_start in range(0, len(requests), args.request_chunk_size):
        chunk = requests[chunk_start : chunk_start + args.request_chunk_size]
        vision_cache = {}
        pending = []
        for request in chunk:
            image_path = request["image_path"]
            if image_path not in vision_cache:
                image_inputs, _, _ = process_vision_info(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": image_path}],
                        }
                    ],
                    image_patch_size=processor.image_processor.patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                vision_cache[image_path] = {"image": image_inputs}

            text = (
                f'Prompt: "{request["prompt"]}"\n'
                f'Question: "{request["question"]}"'
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            pending.append(
                {
                    "request_id": request["request_id"],
                    "input": {
                        "prompt": prompt,
                        "multi_modal_data": vision_cache[image_path],
                    },
                }
            )

        max_tokens = args.initial_max_tokens
        for round_index in range(args.max_rounds):
            if not pending:
                break
            sampling = SamplingParams(
                temperature=0.0,
                repetition_penalty=1.05,
                max_tokens=max_tokens,
            )
            outputs = llm.generate(
                [request["input"] for request in pending],
                sampling_params=sampling,
                use_tqdm=True,
            )
            retry = []
            for request, output in zip(pending, outputs, strict=True):
                raw = output.outputs[0].text
                text = _extract_text(raw, round_index == args.max_rounds - 1)
                if not text and round_index < args.max_rounds - 1:
                    retry.append(request)
                    continue
                responses[request["request_id"]] = {
                    "request_id": request["request_id"],
                    "raw_response": text,
                    "error": None if text else "empty_response",
                }
            pending = retry
            max_tokens = min(max_tokens * 4, 8192)

    with Path(args.output).open("w", encoding="utf-8") as handle:
        for request in requests:
            response = responses.get(
                request["request_id"],
                {
                    "request_id": request["request_id"],
                    "raw_response": "",
                    "error": "missing_response",
                },
            )
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
