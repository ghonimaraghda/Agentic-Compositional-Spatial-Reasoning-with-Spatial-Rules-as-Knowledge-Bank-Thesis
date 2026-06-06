#!/bin/bash
#SBATCH --job-name=sparp_relation_plot
#SBATCH --output=/storage/ukp/work/ghonema/agentic/plot_output.txt
#SBATCH --error=/storage/ukp/work/ghonema/agentic/plot_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

export WORKSPACE_ROOT=/storage/ukp/work/ghonema/agentic
cd "$WORKSPACE_ROOT"

# Use your existing Python environment
ENV_PY=/storage/ukp/work/ghonema/miniconda3/envs/qwen35/bin/python

$ENV_PY image.py