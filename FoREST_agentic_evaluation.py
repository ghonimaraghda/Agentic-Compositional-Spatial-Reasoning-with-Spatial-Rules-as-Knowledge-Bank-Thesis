#!/usr/bin/env python3
# /path/to/forest_qa_cot_vllm_qwen2.py

import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

VALID = {"front", "back", "left", "right"}
LABEL_RE = re.compile(r"\b(front|back|left|right)\b", re.IGNORECASE)
ANSWER_RE = re.compile(r"(?is)\banswer\s*:\s*(front|back|left|right)\b")

# ========= FoREST prompt.py (QA part) =========

QA_prompt_COT = """You are a useful question-answering system, especially in language and spatial relations. You will be given the textual description of the scene and the corresponding question in the format "Context: sentence(s). Question: sentence.” "
Answer the question with a reasonable explanation.
There are four answer candidates {front, back, left, right} indicate the relation asked in the question.  The answer is in the format of "Explanation: sentence(s). Answer: candidate."
"""

QA_COT_ex = [
    {
        "role": "user",
        "content": "Context: The bird is outside the car and in front of the car relative to the car. The car is facing toward the camera. Question: Based on the camera's perspective, where is the bird from the car position in the scene?",
    },
    {
        "role": "assistant",
        "content": "Explanation: Based on the context, the bird's position is in the front direction of the car. The car is facing the camera. Then, the car's front direction is the camera's front direction. Therefore, the bird's position is in front of the car's position from the camera's perspective. Answer: front",
    },
    {
        "role": "user",
        "content": "Context: The bird is inside the car and left of the car from the car's perspective. The car is facing to the right relative to the camera.  Question: Based on the camera's perspective, where is the bird from the car's position?",
    },
    {
        "role": "assistant",
        "content": "Explanation: Based on the context, the bird's position is in the left direction of the car. The car is facing to the right. Then, the car's left direction is the camera's back direction. Therefore, the bird's position is to the back of the car's position from the camera's perspective. Answer: back",
    },
    {
        "role": "user",
        "content": "Context: The box is inside and to the right of the room from the observer's perspective. Question: From the observer's perspective, what is the spatial relation of the box to the room?",
    },
    {
        "role": "assistant",
        "content": "Explanation: Based on the context, the box is to the right of the room from the camera's direction. Therefore, the box's position is to the right of the room's position from the observer's perspective. Answer: right",
    },
    {
        "role": "user",
        "content": "Context: A phone is to the left of a tablet from my perspective. The tablet is facing to the right. Question: From my perspective, what is the spatial relation of the phone to the tablet?",
    },
    {
        "role": "assistant",
        "content": "Explanation: Based on the context, the phone is to the left of the tablet from my perspective. The direction of the tablet is not relevant in this context since the left relation is from my perspective. Therefore, from my perspective, the phone is to the left of the tablet. Answer: left",
    },
]


def normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    t = str(x).strip().lower()
    if t in VALID:
        return t
    if t == "behind":
        return "back"
    if t == "forward":
        return "front"
    return None


def gold_set_from_candidate_answer(example: Dict[str, Any]) -> Set[str]:
    ca = example.get("candidate_answer")
    if isinstance(ca, list):
        out = {normalize_label(v) for v in ca}
        return {v for v in out if v in VALID}
    if isinstance(ca, str):
        v = normalize_label(ca)
        return {v} if v in VALID else set()
    return set()


def extract_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "unknown"

    # Prefer the LAST "Answer: X"
    matches = list(ANSWER_RE.finditer(t))
    if matches:
        return matches[-1].group(1).lower()

    # Fallback to last standalone label token
    labels = LABEL_RE.findall(t)
    if labels:
        return labels[-1].lower()

    low = t.lower()
    if "behind" in low:
        return "back"
    if "forward" in low:
        return "front"

    return "unknown"


def build_chat_messages(context: str, question: str) -> List[Dict[str, str]]:
    # Mirror FoREST: system + few-shot + final user.
    # Note: FoREST uses "Context: {context} Question: {question}" exactly.
    return (
        [{"role": "system", "content": QA_prompt_COT}]
        + list(QA_COT_ex)
        + [{"role": "user", "content": f"Context: {context} Question: {question}"}]
    )


def to_model_prompt(tokenizer: AutoTokenizer, chat_messages: List[Dict[str, str]]) -> str:
    # Important for Qwen2: use chat template; raw string often underperforms.
    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def chunked(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--limit_n", type=int, default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_mem", type=float, default=0.90)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--log_every", type=int, default=20, help="Print running accuracy every N examples.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading vLLM model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
    )

    # FoREST pipeline() doesn’t enforce stops; we keep gentle stops to reduce trailing babble.
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["\n\n", "\n\n\n"],
    )

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)["data"]

    if args.limit_n is not None:
        dataset = dataset[: args.limit_n]
        print(f"DEBUG: limiting to first {args.limit_n} examples")

    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for ex in dataset:
        context = str(ex.get("context", ""))
        question = str(ex.get("question", ""))
        chat = build_chat_messages(context, question)
        prompt = to_model_prompt(tokenizer, chat)
        prompts.append(prompt)
        metas.append(ex)

    results: List[Dict[str, Any]] = []
    correct = 0
    processed = 0
    log_every = max(1, args.log_every)


    for p_batch, m_batch in zip(chunked(prompts, args.batch_size), chunked(metas, args.batch_size)):
        outs = llm.generate(p_batch, sampling)
        for out, ex in zip(outs, m_batch):
            raw = out.outputs[0].text if out.outputs else ""
            pred = extract_answer(raw)

            golds = gold_set_from_candidate_answer(ex)
            is_correct = pred in golds

            correct += int(is_correct)
            results.append(
                {
                    "id": ex.get("id"),
                    "context": ex.get("context"),
                    "question": ex.get("question"),
                    "gold_answers": sorted(list(golds)),
                    "predicted_answer": pred,
                    "correct": is_correct,
                    "raw_output": raw,
                }
            )

            processed += 1
            if processed % log_every == 0:
                running_acc = correct / processed
                print(f"[progress] {processed}/{len(dataset)} | running_acc={running_acc:.4f} ({correct}/{processed})", flush=True)


    acc = correct / len(dataset) if dataset else 0.0
    print("=" * 80)
    print(f"Accuracy: {correct}/{len(dataset)} = {acc:.4f}")
    print("=" * 80)

    out_json = os.path.join(args.outdir, "forest_qa_cot_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "total": len(dataset),
                "correct": correct,
                "model": args.model_path,
                "max_tokens": args.max_tokens,
                "results": results,
            },
            f,
            indent=2,
        )

    out_csv = os.path.join(args.outdir, "forest_qa_cot_results.csv")
    pd.DataFrame(
        [
            {
                "id": r["id"],
                "gold_answers": "|".join(r["gold_answers"]),
                "predicted_answer": r["predicted_answer"],
                "correct": r["correct"],
            }
            for r in results
        ]
    ).to_csv(out_csv, index=False)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
