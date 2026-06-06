"""
Agentic compositional spatial-relation evaluation.
- prompt construction is delegated to agentic_orchestrator_reasoning.py
- parsing and repair are delegated to DisambiguationRepairAgent
- dataset loading, metrics, saving, and loop structure remain the same
"""

import os
import json
import argparse
from typing import List

from datasets import load_dataset
from vllm import LLM, SamplingParams

from agentic_orchestrator_reasoning import (
    AgenticSpatialOrchestrator,
    COMPOSITIONAL_LABELS as ALL_LABELS,
    sort_compositional_labels,
)


LLM_ENGINE = None
SAMPLING_PARAMS = None
ORCHESTRATOR = AgenticSpatialOrchestrator(task="compositional_multilabel")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--output_prefix", type=str, default="compositional_agentic")
    return p.parse_args()


def sort_labels(labels):
    return sort_compositional_labels(labels)


def load_data(args):
    split = args.split
    if args.max_examples:
        split = f"{split}[:{args.max_examples}]"

    return load_dataset(
        "UKPLab/sparp",
        name="small-SpaRP-PS1 (SpaRTUN)",
        split=split
    )


def init_llm(args):
    global LLM_ENGINE, SAMPLING_PARAMS

    LLM_ENGINE = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        max_num_seqs=max(1, args.batch_size),
        max_num_batched_tokens=8192,
        enforce_eager=True,
        disable_log_stats=True,
    )

    SAMPLING_PARAMS = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def run_llm(prompt):
    outputs = LLM_ENGINE.generate([prompt], SAMPLING_PARAMS)
    return outputs[0].outputs[0].text.strip()


def run_llm_batch(prompts: List[str]) -> List[str]:
    outputs = LLM_ENGINE.generate(prompts, SAMPLING_PARAMS)
    return [
        out.outputs[0].text.strip() if out.outputs else ""
        for out in outputs
    ]


def build_prompt(example):
    return ORCHESTRATOR.build_prompt(
        tokenizer=None,
        context=example["context"],
        question=example["question"],
    )


def parse_llm_answer(raw):
    return ORCHESTRATOR.parse_answer(raw)


def repair_answer(raw_answer):
    prompt = ORCHESTRATOR.build_repair_prompt(raw_answer)
    raw = run_llm(prompt)
    return parse_llm_answer(raw), raw


def accuracy(preds, golds):
    return sum(set(p) == set(g) for p, g in zip(preds, golds)) / len(preds)


def f1_score(preds, golds):
    tp = fp = fn = 0

    for pred, gold in zip(preds, golds):
        ps, gs = set(pred), set(gold)

        for label in ALL_LABELS:
            if label in ps and label in gs:
                tp += 1
            elif label in ps and label not in gs:
                fp += 1
            elif label not in ps and label in gs:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def chunked(xs, n):
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def main():
    args = parse_args()

    print("=" * 80)
    print("AGENTIC COMPOSITIONAL SPATIAL RELATION REASONING")
    print("=" * 80)
    print("Agents: ContextRouterAgent -> RuleBankPromptAgent -> DisambiguationRepairAgent")
    print(f"Model       : {args.model_path}")
    print(f"Split       : {args.split}")
    print(f"Max examples: {args.max_examples}")
    print(f"Temperature : {args.temperature}")
    print(f"Max tokens  : {args.max_tokens}")
    print("=" * 80)

    init_llm(args)
    data = load_data(args)

    os.makedirs(args.outdir, exist_ok=True)

    preds, golds, results = [], [], []

    empty_before_repair = 0
    empty_after_repair = 0

    data_list = list(data)
    prompts = [build_prompt(ex) for ex in data_list]

    processed = 0

    for ex_batch, prompt_batch in zip(chunked(data_list, args.batch_size), chunked(prompts, args.batch_size)):
        raws = run_llm_batch(prompt_batch)

        for ex, raw in zip(ex_batch, raws):
            pred = parse_llm_answer(raw)

            repair_raw = None
            if not pred:
                empty_before_repair += 1
                pred, repair_raw = repair_answer(raw)

            if not pred:
                empty_after_repair += 1

            gold = sort_labels(ex["targets"])
            correct = set(pred) == set(gold)

            preds.append(pred)
            golds.append(gold)

            if processed % args.log_every == 0:
                print(f"[{processed}/{len(data_list)}]")
                print(f"Q      : {ex['question']}")
                print(f"Gold   : {gold}")
                print(f"Pred   : {pred}")
                print(f"Correct: {correct}")
                print(f"Raw    : {raw[:300]}")
                print()

            results.append({
                "example": processed,
                "question": ex["question"],
                "context": ex["context"],
                "gold": gold,
                "predicted": pred,
                "correct": correct,
                "target_choices": ex.get("target_choices"),
                "symbolic_question": ex.get("symbolic_question"),
                "symbolic_context": ex.get("symbolic_context"),
                "symbolic_entity_map": ex.get("symbolic_entity_map"),
                "symbolic_reasoning_reference": ex.get("symbolic_reasoning"),
                "agent_trace": {
                    "router_agent": "selected compositional_multilabel rule bank",
                    "prompt_agent": "built prompt with inverse/symmetry/transitivity/combination/not rules",
                    "repair_agent": "parsed JSON list and repaired only if empty or malformed",
                },
                "debug": {
                    "llm_raw": raw,
                    "llm_repair_raw": repair_raw,
                    "prompt_used_test_symbolic_reasoning": False,
                    "prompt_used_test_gold": False,
                    "few_shots_are_fixed": True,
                    "mode": "agentic_fixed_agents_rule_bank_prompting",
                    "rules_used": [
                        "inverse",
                        "symmetry",
                        "transitivity",
                        "combination",
                        "not",
                    ],
                }
            })

            processed += 1

    acc = accuracy(preds, golds)
    f1 = f1_score(preds, golds)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")
    print("-" * 80)
    print(f"Empty before repair: {empty_before_repair}")
    print(f"Empty after repair : {empty_after_repair}")

    out_path = os.path.join(args.outdir, f"{args.output_prefix}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": acc,
                "f1_score": f1,
                "correct": sum(r["correct"] for r in results),
                "total": len(results),
                "empty_before_repair": empty_before_repair,
                "empty_after_repair": empty_after_repair,
            },
            "pipeline": {
                "setup": "agentic fixed-agent rule-bank prompting",
                "dataset": "small-SpaRP-PS1 (SpaRTUN)",
                "model": args.model_path,
                "uses_test_symbolic_reasoning_in_prompt": False,
                "uses_test_gold_in_prompt": False,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "rules": [
                    "Inverse",
                    "Symmetry",
                    "Transitivity",
                    "Combination",
                    "Not",
                ],
                **ORCHESTRATOR.metadata(),
            },
            "results": results,
        }, f, indent=2)

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
