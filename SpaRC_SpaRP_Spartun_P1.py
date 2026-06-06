"""

Fixed 5-shot CoT prompt + LLM prediction 
"""

import os, json, re, argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


ALL_LABELS = [
    "left", "right", "above", "below",
    "behind", "in front", "near", "far",
    "outside", "outside and touching", "partially overlapping",
    "inside and touching", "inside",
    "contains and touches", "contains", "overlapping"
]

#This teaches the model the expected reasoning style and output format from validation split.
FEW_SHOT_BLOCK = """
Example 1:
Context:
There exist three blocks, called AAA, BBB and CCC. Block AAA contains a big blue circle. This block is inside block BBB. A medium yellow circle and a big black circle are inside block BBB. Block BBB covers a medium blue square. The medium yellow circle is behind the big black circle and is above the medium blue square. Block BBB is disconnected from and in front of block CCC. A medium blue triangle is inside block CCC.

Question:
Where is the medium blue triangle regarding the big blue circle?

Reasoning:
Step 1: It is given that the medium blue triangle is inside the block CCC.
Step 2: From the context, the block BBB is outside and in front of the block CCC.
Step 3: From step 2, it can be inferred that the block CCC is outside and behind the block BBB.
Step 4: From step 1 and 3, we can infer that the medium blue triangle is behind and outside the block BBB.
Step 5: From the context, the block AAA is inside the block BBB.
Step 6: From step 5, we can say that the block BBB contains the block AAA.
Step 7: From step 4 and 6, it can be inferred that the medium blue triangle is behind and outside the block AAA.
Step 8: It is given that the block AAA contains the big blue circle.
Step 9: From step 7 and 8, we can say that the medium blue triangle is behind and outside the big blue circle.

Final answer:
[
"behind",
"outside"
]

Example 2:
Context:
Three blocks, named AAA, BBB and CCC exist. Block BBB is in block CCC. Block AAA with a medium yellow square is disconnected from and to the right of block CCC. Block CCC is under this block. A medium blue square is within block AAA. Block BBB covers a medium yellow square. A medium black square is inside this block.
Question:
Where is the medium yellow square in AAA relative to the medium black square?

Reasoning:
Step 1: From the context, the block AAA contains the medium yellow square of block AAA.
Step 2: From step 1, we can say that the medium yellow square of block AAA is inside the block AAA.
Step 3: From the context, the block CCC is below the block AAA.
Step 4: It is given that the block AAA is outside and right of the block CCC.
Step 5: From step 3 and 4, we can say that the block AAA is above, outside and right of the block CCC.
Step 6: From step 2 and 5, we can say that the medium yellow square of block AAA is above, outside and right of the block CCC.
Step 7: It is given that the block BBB is inside the block CCC.
Step 8: From step 7, it can be inferred that the block CCC contains the block BBB.
Step 9: From step 6 and 8, we can infer that the medium yellow square of block AAA is above, outside and right of the block BBB.
Step 10: It is given that the medium black square is inside the block BBB.
Step 11: From step 10, we can infer that the block BBB contains the medium black square.
Step 12: From step 9 and 11, we can say that the medium yellow square of block AAA is above, outside and right of the medium black square.

Final answer:
[
"right",
"above",
"outside"
]

Example 3:
Context:
Three blocks, called HHH, LLL and KKK exist in the image. Block KKK is under and to the right-hand side of block LLL. Block LLL has block HHH. A large purple pentagon is within block HHH. Block HHH cover a large red hexagon and a medium purple hexagon. Under the large purple pentagon is the large red hexagon. Under the medium purple hexagon there is the large purple pentagon. Block LLL has a large purple pentagon. A medium red hexagon and a medium grey pentagon are in block KKK. Block KKK has a large grey pentagon.

Question:
What is the position of the large grey pentagon regarding the large purple pentagon in HHH?

Reasoning:
Step 1: It is given that the block KKK contains the large grey pentagon.
Step 2: From step 1, we can say that the large grey pentagon is inside the block KKK.
Step 3: From the context, the block KKK is below and right of the block LLL.
Step 4: From step 2 and 3, we can say that the large grey pentagon is below and right of the block LLL.
Step 5: From the context, the block LLL contains the block HHH.
Step 6: From step 4 and 5, we can infer that the large grey pentagon is below and right of the block HHH.
Step 7: It is given that the large purple pentagon of block HHH is inside the block HHH.
Step 8: From step 7, we can infer that the block HHH contains the large purple pentagon of block HHH.
Step 9: From step 6 and 8, it can be inferred that the large grey pentagon is below and right of the large purple pentagon of block HHH.

Final answer:
[
"right",
"below"
]

Example 4:
Context:
Three boxes, named DDD, EEE and JJJ exist. Box EEE is at 6:00 position relative to box DDD. This box is at 12:00 position regarding to and in front of box JJJ. In front of this box there is box DDD with a midsize orange rectangle. This box covers a midsize white thing. A midsize green rectangle and another midsize orange rectangle are within this box. At 6:00 position regarding to midsize orange rectangle number two is the white shape. At 6 o'clock position regarding to midsize orange rectangle number one is midsize orange rectangle number two. Midsize orange rectangle number one is at 6:00 position relative to the midsize green rectangle. A midsize orange rectangle is within box JJJ. Box JJJ have a midsize green rectangle and another midsize orange rectangle.
Question:
	
Where is the midsize green rectangle in JJJ relative to the midsize green rectangle in DDD?

Reasoning:
Step 1: It is given that the box JJJ contains the midsize green rectangle of box JJJ.
Step 2: From step 1, we can infer that the midsize green rectangle of box JJJ is inside the box JJJ.
Step 3: It is given that the box EEE is above and in front of the box JJJ.
Step 4: From step 3, we can infer that the box JJJ is below and behind the box EEE.
Step 5: From step 2 and 4, it can be inferred that the midsize green rectangle of box JJJ is below and behind the box EEE.
Step 6: It is given that the box DDD is in front of the box EEE.
Step 7: It is given that the box EEE is below the box DDD.
Step 8: From step 6 and 7, we can infer that the box EEE is behind and below the box DDD.
Step 9: From step 5 and 8, it can be inferred that the midsize green rectangle of box JJJ is below and behind the box DDD.
Step 10: It is given that the midsize green rectangle of box DDD is inside the box DDD.
Step 11: From step 10, it can be inferred that the box DDD contains the midsize green rectangle of box DDD.
Step 12: From step 9 and 11, it can be inferred that the midsize green rectangle of box JJJ is below and behind the midsize green rectangle of box DDD.

Final answer:
[
"below",
"behind"
]
Example 5:
Context:
A small green watermelon and a big orange melon are within a box called one. Box one contains a medium orange melon. Another box named two with a medium orange melon has this box. Disconnected from this box is another box named three with a small orange melon and a medium yellow apple. Box two is far from this box. A small yellow apple is covered by box three.

Question:	
Where is the medium yellow apple regarding the big orange melon?

Reasoning:
Step 1: From the context, the box three contains the medium yellow apple.
Step 2: From step 1, we can infer that the medium yellow apple is inside the box three.
Step 3: It is given that the box two is far from the box three.
Step 4: It is given that the box three is outside the box two.
Step 5: From step 3 and 4, it can be inferred that the box three is outside and far from the box two.
Step 6: From step 2 and 5, it can be inferred that the medium yellow apple is outside and far from the box two.
Step 7: It is given that the box two contains the box one.
Step 8: From step 6 and 7, it can be inferred that the medium yellow apple is outside and far from the box one.
Step 9: From the context, the big orange melon is inside the box one.
Step 10: From step 9, we can say that the box one contains the big orange melon.
Step 11: From step 8 and 10, we can say that the medium yellow apple is outside and far from the big orange melon.

Final answer:
[
"far",
"outside"
]
"""


LLM_ENGINE = None
SAMPLING_PARAMS = None
TOKENIZER = None


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
    return p.parse_args()


def sort_labels(labels):
    s = set(labels)
    return [x for x in ALL_LABELS if x in s]


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
    global LLM_ENGINE, SAMPLING_PARAMS, TOKENIZER

    TOKENIZER = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    LLM_ENGINE = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        enforce_eager=True,
        disable_log_stats=True,
    )

    SAMPLING_PARAMS = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

def to_model_prompt(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_llm(prompt):
    model_prompt = to_model_prompt(prompt)
    outputs = LLM_ENGINE.generate([model_prompt], SAMPLING_PARAMS)
    return outputs[0].outputs[0].text.strip()


def build_prompt(example):
    return f"""
You are an expert assistant with the knowledge of spatial relations and the rules to compose them under the assumptions that the contexts provided are of 'Fixed Orientation Point of View (FPoV)', the objects or entities are to be treated as 'Extended Objects (EO)', the directions are 'Quantitatively Unspecified (QU)', and 'Relation Incomplete (RI)'. The description of these terminologies are as given below:

Fixed Orientation Point of View (FPoV): The spatial relations are expressed from a single, consistent and unchanging perspective. This means that the observations are made from a global viewpoint that remains same and constant for all the entities in a given context. Hence, relations such as relative directions e.g. left or right always refer to the same directions and there is a one-to-one mapping between relative, cardinal and clock-face directions i.e. left is same as west or 9 o'clock position, right is same as east or 3 o'clock position, above is same as north or 12 o'clock position, and below is same as south or 6 o'clock position.

Extended Objects (EO): Entities can be treated as Extended Objects if they have shapes and sizes which are not to be ignored in the context. Because of this, although a larger set of relations is possible between objects but the composition rules can become complex. For example, a cafe and a mall building can be treated as extended objects and the cafe can be a part of i.e. inside the mall building itself.

Quantitatively Unspecified (QU): Spatial relations, such as directions, specified between two objects are said to be quantitatively unspecified if those relations can have a unit of measurement but are not stated in the specified context. The composition of such relations may not be possible even when all the object parameters and the relations between any two objects in a statement are completely known. For example, even with constraints such as objects A, B and C are apples lying in a line, the quantitatively unspecified statements - B is to the left of A, and C is to the right of B - can not lead to any conclusion regarding left, right, or overlapping relationship between A and C.

Relation Incomplete (RI): Relations are incomplete in the context statements if not all the spatial relationships that exist between two objects are stated. In such cases, multiple spatial outline or positioning of the objects are possible, without a single definitive truth. For example, consider the relationship - the fruit F is behind the object O in a 2D plane. Although O is in front of F, their relative position on the horizontal axis is incomplete, and hence, could be left, right or at the same place when considered horizontally.

You need to identify the sub-set of entities from the context that are relevant as well as combine their spatial relations with valid compositions under the above mentioned assumptions to find the spatial relations between the entities in the asked questions. The list of all possible spatial relations are: left, right, above, below, behind, in front, near, far, outside, outside and touching, partially overlapping, inside and touching, inside, contains and touches, contains, and overlapping. Always provide the final answer, only and only, in terms of these spatial relations. Include all the spatial relations that hold true as the answer, in case of multiple correct choices.
Below are worked examples.

{FEW_SHOT_BLOCK}


For the new question, use a short fixed reasoning style like the examples: first identify the target and reference, then build only the shortest relevant chain between them, then compose only the relations on that chain.
Limit the reasoning to at most 6 short steps.
Do not re-read, restart, verify again, or discuss uncertainty at length.
The last reasoning step should filter contradictions and uncertain labels.
After the reasoning, immediately output the final answer as a valid JSON list.
Now solve the new question.

Context:
{example["context"]}

Question:
{example["question"]}

After the reasoning, output the final answer as a valid JSON list using only the allowed labels.

Final answer:
""".strip()


def parse_llm_answer(raw):
    matches = re.findall(r"\[.*?\]", raw, flags=re.DOTALL)

    for m in reversed(matches):
        try:
            arr = json.loads(m)
        except Exception:
            continue

        if not isinstance(arr, list):
            continue

        cleaned = []
        for x in arr:
            if isinstance(x, str):
                x = x.strip().lower()
                if x in ALL_LABELS:
                    cleaned.append(x)

        if cleaned:
            return sort_labels(cleaned)

    return []


def repair_answer(raw_answer):
    prompt = f"""
Convert the following model output into a valid JSON list of spatial relation labels.

Allowed labels:
{ALL_LABELS}

Model output:
{raw_answer}

Rules:
- Use only labels from the allowed labels.
- Return only a JSON list.
- No explanation.

JSON:
""".strip()

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


def main():
    args = parse_args()

    print("=" * 80)
    print("SPARP-S-PS1 FIXED 5-SHOT COT EVALUATION")
    print("=" * 80)
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

    for i, ex in enumerate(data):
        prompt = build_prompt(ex)

        raw = run_llm(prompt)
        pred = parse_llm_answer(raw)

        repair_raw = None
        if not pred:
            pred, repair_raw = repair_answer(raw)

        gold = sort_labels(ex["targets"])
        correct = set(pred) == set(gold)

        preds.append(pred)
        golds.append(gold)

        if i % args.log_every == 0:
            print(f"[{i}/{len(data)}]")
            print(f"Q      : {ex['question']}")
            print(f"Gold   : {gold}")
            print(f"Pred   : {pred}")
            print(f"Correct: {correct}")
            print(f"Raw    : {raw[:300]}")
            print()

        results.append({
            "example": i,
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
            "debug": {
                "llm_raw": raw,
                "llm_repair_raw": repair_raw,
                "prompt_used_test_symbolic_reasoning": False,
                "prompt_used_test_gold": False,
                "few_shots_are_fixed": True,
                "mode": "fixed_5shot_cot"
            }
        })

    acc = accuracy(preds, golds)
    f1 = f1_score(preds, golds)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")

    out_path = os.path.join(args.outdir, "spartun_P1_Qwen2.5-72B-Instruct_reasoning.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": acc,
                "f1_score": f1,
                "correct": sum(r["correct"] for r in results),
                "total": len(results),
            },
            "pipeline": {
                "dataset": "small-SpaRP-PS1 (SpaRTUN)",
                "model": args.model_path,
                "setup": "fixed paper-aligned 5-shot CoT",
                "uses_test_symbolic_reasoning_in_prompt": False,
                "uses_test_gold_in_prompt": False,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
            "results": results,
        }, f, indent=2)

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()