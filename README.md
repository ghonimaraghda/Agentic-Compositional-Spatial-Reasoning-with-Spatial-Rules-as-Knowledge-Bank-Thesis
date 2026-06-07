````markdown
# Agentic Compositional Spatial Reasoning with Spatial Rules as a Knowledge Bank

This repository contains the code and experimental outputs for the master's thesis:

**Agentic Compositional Spatial Reasoning with Spatial Rules as a Knowledge Bank**

The project evaluates whether explicit natural-language spatial rules, organized through an agentic prompting pipeline, can improve large language model performance on textual spatial reasoning tasks.

## Overview

The experiments focus on two spatial reasoning benchmarks:

- **FoREST**: frame-of-reference reasoning and perspective conversion.
- **SpaRP-PS1**: compositional multi-label spatial relation reasoning.

The repository includes baseline prompting scripts, rule-bank/agentic prompting scripts, evaluation outputs, and plotting utilities used to generate the thesis figures.

## Repository Structure

```text
.
├── Images/                              # Generated figures used in the thesis
├── results/                             # Model outputs and evaluation results
├── C-split_QA_camera_total.json         # FoREST camera-perspective QA data
├── C-split_QA_relatum_total.json        # FoREST relatum-perspective QA data
├── split_QA_camera_total_reverse.json   # FoREST auxiliary/reverse camera data
├── FoREST_P1.py                         # FoREST baseline prompt evaluation
├── forest_agentic_reasoning.py          # FoREST rule-bank / agentic evaluation
├── FoREST_agentic_evaluation.py         # FoREST agentic evaluation utility
├── SpaRC_SpaRP_Spartun_P1.py            # SpaRP-PS1 baseline prompt evaluation
├── spartun_agentic_reasoning.py         # SpaRP-PS1 rule-bank / agentic evaluation
├── agentic_orchestrator_reasoning.py    # Shared agentic orchestration logic
├── image.py                             # Plot generation script
├── image.sh                             # SLURM script for plot generation
├── run.sh                               # SLURM/run script for baseline experiments
├── run_agentic.sh                       # SLURM/run script for agentic experiments
├── requirements.txt                     # Python dependencies
├── res.txt                              # Runtime/result log file
└── README.md
````

## Method

The proposed pipeline separates spatial reasoning into three stages:

1. **ContextRouterAgent**
   Selects the appropriate reasoning profile for the input task.

2. **RuleBankPromptAgent**
   Constructs the prompt using task-specific natural-language spatial rules.

3. **ValidationAgent**
   Normalizes and validates the model output before evaluation.

For **FoREST**, the rule bank focuses on frame-of-reference categories and direction conversion rules.

For **SpaRP-PS1**, the rule bank focuses on spatial relation groups, inverse rules, symmetry rules, transitivity, containment propagation, and contradiction filtering.

## Models

The experiments were conducted using open-weight instruction-tuned LLMs, including:

* `Meta-Llama-3-70B-Instruct`
* `Qwen2.5-72B-Instruct`
* `Qwen3.5-27B`

Inference was run with `vLLM` on a SLURM-managed HPC environment.

## Running Experiments

Example FoREST baseline run:

```bash
srun $ENV_PY $WORKSPACE_ROOT/FoREST_P1.py \
  --model_path /path/to/model \
  --tp 2 \
  --max_tokens 768 \
  --outdir $WORKSPACE_ROOT/results \
  --log_every 20
```

Example SpaRP-PS1 baseline run:

```bash
srun $ENV_PY $WORKSPACE_ROOT/SpaRC_SpaRP_Spartun_P1.py \
  --model_path /path/to/model \
  --tp 2 \
  --max_tokens 768 \
  --outdir $WORKSPACE_ROOT/results \
  --log_every 20
```

To run only a limited number of examples, use:

```bash
--max_examples 20
```

The included shell scripts can also be used as templates:

```bash
bash run.sh
bash run_agentic.sh
```

or, on SLURM:

```bash
sbatch run.sh
sbatch run_agentic.sh
```

## Evaluation Metrics

### FoREST

FoREST is evaluated using:

* **Micro-averaged accuracy**
* **Frame-of-reference category-level accuracy**

The answer space is restricted to:

```text
front, back, left, right
```

### SpaRP-PS1

SpaRP-PS1 is evaluated using:

* **Exact-match accuracy**
* **Macro-averaged F1**
...
Macro-averaged F1 measures partial relation recovery by computing F1 per relation label and averaging across labels.
Macro-averaged F1 measures partial relation recovery across all labels.

## Plot Generation

The plotting script can be run directly:

```bash
python image.py
```

or through SLURM:

```bash
sbatch image.sh
```

Generated figures are saved in the `Images/` directory.

## Notes

* The experiments are inference-time prompting experiments.
* No model fine-tuning was performed.
* No external symbolic solver was used.
* Benchmark-provided symbolic reasoning traces were not exposed to the models during inference.
* The `results/` directory contains generated outputs used for evaluation and figure generation.

## Citation

If you use or refer to this repository, please cite:

```text
Raghda Ghonema. Agentic Compositional Spatial Reasoning with Spatial Rules as a Knowledge Bank. Master's Thesis, TU Darmstadt, 2026.
```

```
```
