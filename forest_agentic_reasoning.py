"""
Agentic directional frame-of-reference evaluation.

This is your original FoREST-style file with minimal changes:
- prompt construction is delegated to agentic_orchestrator_reasoning.py
- answer parsing / gold normalization is delegated to the repair agent
- evaluation, batching, metrics, and saving remain the same
"""

import os
import json
import csv
import argparse
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agentic_orchestrator_reasoning import AgenticSpatialOrchestrator


VALID_CATEGORIES = {
    "external relative",
    "external intrinsic",
    "internal intrinsic",
    "internal relative",
}


def get_for_category(example: Dict[str, Any]) -> str:
    label = example.get("label")

    if isinstance(label, list) and label:
        text = str(label[0]).strip().lower()
        return text if text in VALID_CATEGORIES else "unknown"

    if isinstance(label, str):
        text = label.strip().lower()
        return text if text in VALID_CATEGORIES else "unknown"

    return "unknown"


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
    agentic_metadata: Dict[str, Any],
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
                "pipeline": {
                    "setup": "agentic directional frame-of-reference reasoning",
                    **agentic_metadata,
                },
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
    parser.add_argument("--max_model_len", type=int, default=3072)
    parser.add_argument("--gpu_mem", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--output_prefix", type=str, default="directional_agentic")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    orchestrator = AgenticSpatialOrchestrator(task="directional_frame")

    print("=" * 80)
    print("AGENTIC DIRECTIONAL FRAME-OF-REFERENCE REASONING")
    print("=" * 80)
    print("Agents: ContextRouterAgent -> RuleBankPromptAgent -> DisambiguationRepairAgent")
    print(f"Model : {args.model_path}")
    print("=" * 80)

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
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            terminators.append(eot_id)
    except Exception:
        pass

    sampling = SamplingParams(
        temperature=1e-8,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop_token_ids=terminators,
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

        prompt = orchestrator.build_prompt(tokenizer, context, question)

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

    final_json = os.path.join(args.outdir, f"{args.output_prefix}.json")
    final_csv = os.path.join(args.outdir, f"{args.output_prefix}.csv")

    for p_batch, m_batch in zip(chunked(prompts, args.batch_size), chunked(metas, args.batch_size)):
        outs = llm.generate(p_batch, sampling)

        for out, ex in zip(outs, m_batch):
            raw = out.outputs[0].text if out.outputs else ""
            pred = orchestrator.parse_answer(raw)

            golds = orchestrator.repair_agent.gold_directional_set(ex)
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
                    "agent_trace": {
                        "router_agent": "selected directional_frame rule bank",
                        "prompt_agent": "built chat prompt with frame-of-reference rules",
                        "repair_agent": "extracted one label from allowed labels",
                    },
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
        orchestrator.metadata(),
    )

    print(f"Saved: {final_json}")
    print(f"Saved: {final_csv}")


if __name__ == "__main__":
    main()
