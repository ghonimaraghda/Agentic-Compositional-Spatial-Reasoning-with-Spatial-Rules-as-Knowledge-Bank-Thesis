import os
import json
import re
import csv
import argparse
from typing import Any, Dict, List, Optional, Set

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

VALID = {"front", "back", "left", "right"}
LABEL_RE = re.compile(r"\b(front|back|left|right)\b", re.IGNORECASE)
ANSWER_RE = re.compile(r"(?is)\banswer\s*:\s*(front|back|left|right)\b")

# ========= FoREST prompt.py (QA SG+CoT part) =========

QA_SG_COT = """You are a useful question-answering system, especially in language and spatial relations. You will be given the textual description of the scene and the corresponding question in the format "Context: sentence(s). Question: sentence.”
Answer the question with a reasonable explanation. Reason briefly and carefully once; do not repeat, restart, or re-check the same conclusion multiple times.
Also, you need to consider the information regrading topological, distance, direction, and frame of reference category.
There are four classes of the frame of reference (external intrinsic, internal intrinsic, external relative, internal relative). Note that the intrinsic direction refers to whether the model has the front/back by itself. (Example: a bird, human. Counter Example: a ball, a box).
External intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one.
Internal intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one.
External relative. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera.
Internal relative. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera.
There are four answer candidates {front, back, left, right} indicating the relation asked in the question. The answer must be in the format: "Explanation: sentence(s). Answer: candidate." Keep the explanation concise and do not repeat the same point."
"""

QA_SG_COT_ex = [
    {"role": "user",
     "content": "Context: The bird is outside the car and in front of the car relative to the car. The car is facing toward the camera. Topological: The bird is outside the car. Distance: The bird is in front of the car. Direction: The bird is in the front direction of the car. Frame of Reference: external intrinsic. Question: Based on the camera's perspective, where is the bird from the car position in the scene?"},
    {"role": "assistant",
     "content": "Explanation: The context states that the bird is in front of the car relative to the car. The car is facing toward the camera, so the car's front direction aligns with the camera's front direction. Therefore, from the camera's perspective, the bird is located in front of the car. Answer: front"},
    {"role": "user",
     "content": "Context: The bird is inside the car and left of the car from the car's perspective. The car is facing to the right relative to the camera. Topological: The bird is inside the car. Distance: The bird is close to the car. Direction: The bird is in the left direction of the car. Frame of Reference: internal intrinsic. Question: Based on the camera's perspective, where is the bird from the car's position?"},
    {"role": "assistant",
     "content": "Explanation: The bird is to the left of the car from the car's perspective. The car is facing to the right relative to the camera. When the car faces right, the car's left direction corresponds to the camera's back direction. Therefore, from the camera's perspective, the bird is behind the car. Answer: back"},
    {"role": "user",
     "content": "Context: The box is inside and to the right of the room from the observer's perspective. Topological: The box is inside the room. Distance: The box is close to the room. Direction: From the observer's perspective, the box is located to the right of the room. Frame of Reference: internal relative. Question: From the observer's perspective, what is the spatial relation of the box to the room?"},
    {"role": "assistant",
     "content": "Explanation: The context states that the box is to the right of the room from the observer's perspective. Because the relation is already defined in the observer's frame of reference, no rotation or conversion is required. Therefore, the box is to the right of the room from the observer's perspective. Answer: right"},
    {"role": "user",
     "content": "Context: A phone is to the left of a tablet from my perspective. The tablet is facing to the right. Topological: The phone is not contained by the tablet. Distance: There is some distance between the phone and the tablet. Direction: From the observer's perspective, the phone is located to the left of the tablet. Frame of Reference: external relative. Question: From my perspective, what is the spatial relation of the phone to the tablet?"},
    {"role": "assistant",
     "content": "Explanation: The context explicitly states that the phone is to the left of the tablet from the observer's perspective. Since the relation is already given in the observer's frame of reference, the facing direction of the tablet does not affect the relation. Therefore, the phone is to the left of the tablet from the observer's perspective. Answer: left"},
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
    """
    Uses the FIRST generated Answer: label.
    This avoids accidentally picking later Answer: labels if the model keeps rambling
    or starts referencing examples after the actual answer.
    """
    t = (text or "").strip()

    matches = list(ANSWER_RE.finditer(t))
    if matches:
        return matches[0].group(1).lower()

    labels = LABEL_RE.findall(t)
    if labels:
        return labels[-1].lower()

    low = t.lower()
    if "behind" in low:
        return "back"
    if "forward" in low:
        return "front"

    return "unknown"


def get_for_category(example: Dict[str, Any]) -> str:
    """
    Uses the original dataset label field, e.g.
    "label": ["internal relative"]
    """

    valid_categories = {
        "external relative",
        "external intrinsic",
        "internal intrinsic",
        "internal relative",
    }

    label = example.get("label")

    if isinstance(label, list) and label:
        text = str(label[0]).strip().lower()
        return text if text in valid_categories else "unknown"

    if isinstance(label, str):
        text = label.strip().lower()
        return text if text in valid_categories else "unknown"

    return "unknown"


def build_chat_messages(context: str, question: str) -> List[Dict[str, str]]:
    user_content = f"Context: {context} Question: {question}\n/no_think"

    return (
        [{"role": "system", "content": QA_SG_COT}]
        + list(QA_SG_COT_ex)
        + [{"role": "user", "content": user_content}]
    )


def to_model_prompt(tokenizer: AutoTokenizer, chat_messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def chunked(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def compute_category_accuracy(category_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    category_accuracy = {}

    for cat, stats in category_stats.items():
        total = stats["total"]
        corr = stats["correct"]
        category_accuracy[cat] = {
            "accuracy": corr / total if total else 0.0,
            "correct": corr,
            "total": total,
        }

    return category_accuracy


def save_outputs(
    out_json: str,
    out_csv: str,
    results: List[Dict[str, Any]],
    accuracy: float,
    category_accuracy: Dict[str, Dict[str, float]],
    total: int,
    correct: int,
    model: str,
    max_tokens: int,
) -> None:
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "category_accuracy": category_accuracy,
                "total": total,
                "correct": correct,
                "model": model,
                "max_tokens": max_tokens,
                "results": results,
            },
            f,
            indent=2,
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "frame_of_reference_category",
            "gold_answers",
            "predicted_answer",
            "correct",
        ])

        for r in results:
            writer.writerow([
                r["id"],
                r["frame_of_reference_category"],
                "|".join(r["gold_answers"]),
                r["predicted_answer"],
                r["correct"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--limit_n", type=int, default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--log_every", type=int, default=20)
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
        enforce_eager=True,
)

    terminators = [tokenizer.eos_token_id]

    for tok in ["<|eot_id|>", "<|im_end|>"]:
         try:
             tok_id = tokenizer.convert_tokens_to_ids(tok)
             if isinstance(tok_id, int) and tok_id >= 0:
                terminators.append(tok_id)
         except Exception:
             pass 

    sampling = SamplingParams(
        temperature=1e-8,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop_token_ids=terminators,
        stop=["\n\nWait", "\nWait", "Let's check again", "Let's re-check"],
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

    category_stats = {
        "external relative": {"correct": 0, "total": 0},
        "external intrinsic": {"correct": 0, "total": 0},
        "internal intrinsic": {"correct": 0, "total": 0},
        "internal relative": {"correct": 0, "total": 0},
        "unknown": {"correct": 0, "total": 0},
    }

    final_json = os.path.join(args.outdir, "forest_P1_Qwen3.5-27B_camera_reasoning.json")
    final_csv = os.path.join(args.outdir, "forest_P1_Qwen3.5-27B_camera_reasoning.csv")

    for p_batch, m_batch in zip(chunked(prompts, args.batch_size), chunked(metas, args.batch_size)):
        outs = llm.generate(p_batch, sampling)

        for out, ex in zip(outs, m_batch):
            raw = out.outputs[0].text if out.outputs else ""
            pred = extract_answer(raw)

            golds = gold_set_from_candidate_answer(ex)
            is_correct = pred in golds

            for_category = get_for_category(ex)
            if for_category not in category_stats:
                for_category = "unknown"

            category_stats[for_category]["total"] += 1
            category_stats[for_category]["correct"] += int(is_correct)

            correct += int(is_correct)
            processed += 1

            results.append(
                {
                    "id": ex.get("id"),
                    "context": ex.get("context"),
                    "question": ex.get("question"),
                    "frame_of_reference_category": for_category,
                    "gold_answers": sorted(list(golds)),
                    "predicted_answer": pred,
                    "correct": is_correct,
                    "raw_output": raw,
                }
            )

            if processed % max(1, args.log_every) == 0:
                running_acc = correct / processed
                running_category_accuracy = compute_category_accuracy(category_stats)

                print(
                    f"[progress] {processed}/{len(dataset)} | running_acc={running_acc:.4f} ({correct}/{processed})",
                    flush=True,
                )

                for cat, stats in running_category_accuracy.items():
                    if stats["total"] > 0:
                        print(
                            f"  {cat:20s}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}",
                            flush=True,
                        )

    acc = correct / len(dataset) if dataset else 0.0
    category_accuracy = compute_category_accuracy(category_stats)

    print("=" * 80)
    print(f"Overall Accuracy: {correct}/{len(dataset)} = {acc:.4f}")
    print("=" * 80)
    print("Accuracy by Frame of Reference:")
    for cat, stats in category_accuracy.items():
        print(f"{cat:20s}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
    print("=" * 80)

    save_outputs(
        final_json,
        final_csv,
        results,
        acc,
        category_accuracy,
        len(dataset),
        correct,
        args.model_path,
        args.max_tokens,
    )

    print(f"Saved: {final_json}")
    print(f"Saved: {final_csv}")


if __name__ == "__main__":
    main()