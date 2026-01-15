import os
import json
import re
import math
from datasets import load_dataset
from vllm import LLM, SamplingParams

# -----------------------------
# Environment & Model Setup
# -----------------------------
os.environ["HF_DATASETS_CACHE"] = "/scratch/$USER/hf_datasets"

MODEL_PATH = "/storage/ukp/shared/shared_model_weights/models--Qwen3-8B"
#change this model Qwen2.5-7B-Instruct to Qwen3-8B
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
)

def run_llm(prompt):
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()

# ============================================================
# SIMPLIFIED TRAVERSAL AGENT
# ============================================================
def build_simple_traversal_prompt(example):
    """Simplified prompt for relation extraction"""
    
    context = example['context']
    question = example['question']
    
    # Clean context - take first few sentences
    sentences = context.split('. ')
    if len(sentences) > 5:
        context = '. '.join(sentences[:5]) + "..."
    
    return f"""EXTRACT SPATIAL RELATIONS FROM CONTEXT

CONTEXT:
{context}

QUESTION:
{question}

TASK:
Extract all spatial relations between entities. Each relation should include:
1. First entity
2. Direction (above, below, left, right)
3. Second entity
4. Unit distance (default is 1)

FORMAT:
[["Entity1", "direction", "Entity2", 1], ["Entity3", "direction", "Entity4", 1]]

EXAMPLES:
Input: "A is above B. B is left of C."
Output: [["A", "above", "B", 1], ["B", "left", "C", 1]]

Input: "X is right of Y. Y is below Z."
Output: [["X", "right", "Y", 1], ["Y", "below", "Z", 1]]

Now extract relations from the given context.

Output ONLY the JSON list, nothing else.
"""

def extract_simple_relations(example):
    """Extract relations from context"""
    prompt = build_simple_traversal_prompt(example)
    raw_text = run_llm(prompt)
    
    print(f"\n[DEBUG] Traversal Agent Raw Output:\n{raw_text[:300]}...")
    
    # Clean the text - remove markdown, extra text
    clean_text = raw_text.strip()
    
    # Try to find JSON array
    json_pattern = r'\[.*\]'
    match = re.search(json_pattern, clean_text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            relations = json.loads(json_str)
            print(f"[DEBUG] Successfully parsed {len(relations)} relations")
            return relations
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parse error: {e}")
    
    # Fallback: manual parsing
    print("[DEBUG] Using fallback parsing...")
    return parse_relations_fallback(clean_text)

def parse_relations_fallback(text):
    """Fallback parsing for relations"""
    relations = []
    
    # Common patterns in StepGame dataset
    patterns = [
        # Simple patterns
        (r'([A-Z])\s+is\s+above\s+([A-Z])', 'above'),
        (r'([A-Z])\s+is\s+below\s+([A-Z])', 'below'),
        (r'([A-Z])\s+is\s+left\s+of\s+([A-Z])', 'left'),
        (r'([A-Z])\s+is\s+right\s+of\s+([A-Z])', 'right'),
        
        # With "the agent" prefix
        (r'the\s+agent\s+([A-Z])\s+is\s+above\s+the\s+agent\s+([A-Z])', 'above'),
        (r'the\s+agent\s+([A-Z])\s+is\s+below\s+the\s+agent\s+([A-Z])', 'below'),
        
        # Compound relations
        (r'([A-Z])\s+is\s+above\s+and\s+left\s+of\s+([A-Z])', [('above', 1), ('left', 1)]),
        (r'([A-Z])\s+is\s+below\s+and\s+right\s+of\s+([A-Z])', [('below', 1), ('right', 1)]),
    ]
    
    for pattern, direction in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(direction, list):
                # Compound relation
                e1, e2 = match.group(1), match.group(2)
                for dir_type, unit in direction:
                    relations.append([e1, dir_type, e2, unit])
            else:
                # Simple relation
                e1, e2 = match.group(1), match.group(2)
                relations.append([e1, direction, e2, 1])
    
    return relations

# ============================================================
# SIMPLIFIED RULE-BASED AGENT (ALGORITHM 1)
# ============================================================
def build_algorithm1_prompt(relations, head, tail):
    """Simple Algorithm 1 prompt"""
    
    if not relations:
        return "No relations to process."
    
    relations_str = ",\n".join([f'  {rel}' for rel in relations])
    
    return f"""APPLY ALGORITHM 1 TO FIND RELATION FROM {head} TO {tail}

RELATIONS:
[
{relations_str}
]

ALGORITHM 1 STEPS:
1. Start with dx=0, dy=0
2. For each relation [entity1, direction, entity2, unit]:
   - If direction is "right": dx = dx + unit
   - If direction is "left": dx = dx - unit
   - If direction is "above": dy = dy + unit
   - If direction is "below": dy = dy - unit
3. Final dx and dy give the relation:
   - dx > 0: right, dx < 0: left
   - dy > 0: above, dy < 0: below

EXAMPLES:
Input: [["A", "above", "B", 1], ["B", "left", "C", 1]]
Calculation: 
  Start: dx=0, dy=0
  Step1: A→B above: dy=0+1=1
  Step2: B→C left: dx=0-1=-1
Result: dx=-1, dy=1 → Answer: ["left", "above"]

Input: [["X", "right", "Y", 2], ["Y", "below", "Z", 1]]
Calculation:
  Start: dx=0, dy=0
  Step1: X→Y right: dx=0+2=2
  Step2: Y→Z below: dy=0-1=-1
Result: dx=2, dy=-1 → Answer: ["right", "below"]

NOW CALCULATE FOR {head} → {tail}:

Step-by-step calculation:
"""

def apply_algorithm1(relations, head, tail):
    """Apply Algorithm 1 with LLM"""
    
    if not relations:
        return {"dx": 0, "dy": 0, "answer": [], "steps": []}
    
    prompt = build_algorithm1_prompt(relations, head, tail)
    raw_text = run_llm(prompt)
    
    print(f"\n[DEBUG] Rule-Based Agent Raw Output:\n{raw_text[:300]}...")
    
    # Try to extract answer from text
    answer = extract_answer_from_text(raw_text)
    
    # If no answer found, compute manually
    if not answer:
        answer = compute_algorithm1_manual(relations)
    
    return answer

def extract_answer_from_text(text):
    """Try to extract answer from LLM output"""
    
    # Look for answer patterns
    answer_patterns = [
        r'Answer:\s*\[(.*?)\]',
        r'Result:\s*\[(.*?)\]',
        r'answer\s*=\s*\[(.*?)\]',
        r'\[(.*?)\]',  # Any JSON-like array
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer_str = match.group(1)
            # Clean and parse
            answer_str = answer_str.replace('"', '').replace("'", "")
            items = [item.strip().lower() for item in answer_str.split(',')]
            items = [item for item in items if item in ['above', 'below', 'left', 'right', 'overlapping']]
            
            if items:
                # Compute dx, dy from answer
                dx, dy = 0, 0
                for item in items:
                    if item == 'right':
                        dx = 1
                    elif item == 'left':
                        dx = -1
                    elif item == 'above':
                        dy = 1
                    elif item == 'below':
                        dy = -1
                
                return {
                    "dx": dx,
                    "dy": dy,
                    "answer": items,
                    "steps": [{"description": "Extracted from LLM output"}]
                }
    
    return None

def compute_algorithm1_manual(relations):
    """Manual computation of Algorithm 1"""
    dx_total = 0
    dy_total = 0
    steps = []
    
    for i, rel in enumerate(relations):
        if len(rel) >= 4:
            e1, direction, e2, unit = rel[0], rel[1], rel[2], rel[3]
            
            dx_change = 0
            dy_change = 0
            
            direction = direction.lower()
            if direction == "right":
                dx_change = unit
            elif direction == "left":
                dx_change = -unit
            elif direction == "above":
                dy_change = unit
            elif direction == "below":
                dy_change = -unit
            
            dx_total += dx_change
            dy_total += dy_change
            
            steps.append({
                "step": i + 1,
                "relation": f"{e1} → {e2}: {direction}",
                "dx_change": dx_change,
                "dy_change": dy_change,
                "dx_total": dx_total,
                "dy_total": dy_total
            })
    
    # Determine answer from dx, dy
    answer = []
    if dx_total > 0:
        answer.append("right")
    elif dx_total < 0:
        answer.append("left")
    
    if dy_total > 0:
        answer.append("above")
    elif dy_total < 0:
        answer.append("below")
    
    if dx_total == 0 and dy_total == 0:
        answer.append("overlapping")
    
    return {
        "dx": dx_total,
        "dy": dy_total,
        "answer": answer,
        "steps": steps
    }

# ============================================================
# ENTITY EXTRACTION
# ============================================================
def extract_entities_from_question(question):
    """Extract head and tail entities from question"""
    
    # Common patterns in StepGame questions
    patterns = [
        r'relation of (?:the )?(?:agent )?([A-Z]) to (?:the )?(?:agent )?([A-Z])',
        r'relation of ([A-Z]) to ([A-Z])',
        r'([A-Z]) to ([A-Z])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
    
    # Fallback: look for capital letters
    entities = re.findall(r'\b[A-Z]\b', question)
    if len(entities) >= 2:
        return entities[0], entities[1]
    
    return "A", "B"  # Default

# ============================================================
# EVALUATION FUNCTIONS 
# ============================================================
def calculate_accuracy(predictions, gold_answers):
    """Calculate accuracy"""
    if not predictions:
        return 0.0
    
    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        # Convert to sets for comparison
        pred_set = set(pred)
        gold_set = set(gold)
        if pred_set == gold_set:
            correct += 1
    
    return correct / len(predictions)

def calculate_f1_score(predictions, gold_answers):
    """Calculate F1 score without sklearn"""
    if not predictions:
        return 0.0
    
    # All possible labels
    all_labels = ["above", "below", "left", "right", "overlapping"]
    
    tp_total = 0  # True positives
    fp_total = 0  # False positives
    fn_total = 0  # False negatives
    
    for pred, gold in zip(predictions, gold_answers):
        pred_set = set(pred)
        gold_set = set(gold)
        
        for label in all_labels:
            pred_has = label in pred_set
            gold_has = label in gold_set
            
            if pred_has and gold_has:
                tp_total += 1
            elif pred_has and not gold_has:
                fp_total += 1
            elif not pred_has and gold_has:
                fn_total += 1
    
    # Calculate precision, recall, F1
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(example):
    """Run the complete pipeline on one example"""
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {example['question']}")
    
    # Extract entities
    head, tail = extract_entities_from_question(example['question'])
    print(f"Head: {head}, Tail: {tail}")
    
    # Step 1: Traversal Agent - Extract relations
    print("\n[Traversal Agent] Extracting relations...")
    relations = extract_simple_relations(example)
    print(f"Extracted {len(relations)} relations")
    
    # Step 2: Rule-Based Agent - Apply Algorithm 1
    print("\n[Rule-Based Agent] Applying Algorithm 1...")
    result = apply_algorithm1(relations, head, tail)
    
    predicted = result.get("answer", [])
    dx = result.get("dx", 0)
    dy = result.get("dy", 0)
    
    print(f"dx={dx}, dy={dy}")
    print(f"Predicted Answer: {predicted}")
    
    return {
        "predicted": predicted,
        "relations": relations,
        "dx": dx,
        "dy": dy,
        "head": head,
        "tail": tail
    }

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """Main function"""
    
    print("=" * 80)
    print("AGENTIC SPATIAL REASONING PIPELINE")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        # Try specific StepGame dataset
        dataset = load_dataset(
            "UKPLab/sparp",
            name="small-SpaRP-PS2 (StepGame)",
            split="test[:5]"
        )
    except Exception as e:
        print(f"Error loading specific dataset: {e}")
        print("Trying generic load...")
        try:
            dataset = load_dataset("UKPLab/sparp", split="test[:5]")
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            # Create dummy dataset for testing
            dataset = [
                {
                    "context": "A is above B. B is left of C.",
                    "question": "What is the relation of A to C?",
                    "targets": ["above", "left"]
                },
                {
                    "context": "X is right of Y. Y is below Z.",
                    "question": "What is the relation of X to Z?",
                    "targets": ["right", "below"]
                }
            ]
    
    all_predictions = []
    all_gold = []
    results = []
    
    print(f"\nProcessing {len(dataset)} examples...")
    
    for i, example in enumerate(dataset):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}")
        print(f"{'='*60}")
        
        # Get gold answer
        if isinstance(example, dict) and 'targets' in example:
            gold = example['targets']
        else:
            gold = ["above", "left"] if i == 0 else ["right", "below"]
        
        print(f"Gold Answer: {gold}")
        
        # Run pipeline
        result = run_pipeline(example)
        
        predicted = result['predicted']
        relations = result['relations']
        
        print(f"Extracted Relations: {relations}")
        print(f"Predicted: {predicted}")
        
        # Check if correct
        is_correct = set(predicted) == set(gold)
        print(f"Correct: {is_correct}")
        
        # Store results
        all_predictions.append(predicted)
        all_gold.append(gold)
        
        results.append({
            "example": i,
            "question": example['question'] if isinstance(example, dict) else "Test question",
            "gold": gold,
            "predicted": predicted,
            "relations": relations,
            "dx": result['dx'],
            "dy": result['dy'],
            "correct": is_correct
        })
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    
    accuracy = calculate_accuracy(all_predictions, all_gold)
    f1 = calculate_f1_score(all_predictions, all_gold)
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"\nAccuracy: {correct_count}/{total_count} = {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save results
    output_file = "spatial_reasoning_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy,
                "f1_score": f1,
                "correct": correct_count,
                "total": total_count
            },
            "results": results,
            "pipeline": {
                "agents": ["Traversal Agent", "Rule-Based Agent (Algorithm 1)"],
                "model": MODEL_PATH
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    main()