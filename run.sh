#!/bin/bash
#SBATCH --job-name=agentic_run
#SBATCH --output=/storage/ukp/work/ghonema/agentic/res.txt
#SBATCH --mail-user=ghonimaraghda@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_mem:80gb"

set -euo pipefail

export WORKSPACE_ROOT=/storage/ukp/work/ghonema/agentic
export HOME=/storage/ukp/work/ghonema

# caches
export HF_HOME=$HOME/.cache/.hf_cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_DATASETS_CACHE"

export VLLM_DISABLE_USAGE_COLLECTION=1
export VLLM_CACHE_DIR="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONPATH="$WORKSPACE_ROOT:${PYTHONPATH:-}"

ENV_PY=/storage/ukp/work/ghonema/miniconda3/envs/thesis310/bin/python

srun $ENV_PY $WORKSPACE_ROOT/FoREST_agentic_evaluation.py \
  --model_path /storage/ukp/shared/shared_model_weights/models--meta-llama--Meta-Llama-3-70B-Instruct \
  --dataset_path /ukp-storage-1/ghonema/agentic/C-split_QA_camera_total.json \
  --tp 2 \
  --max_tokens 256 \
  --batch_size 32 \
  --outdir $WORKSPACE_ROOT/results \
  --log_every 20 
