#!/bin/bash
#
#SBATCH --job-name=agentic_run
#SBATCH --output=/ukp-storage-1/ghonema/agentic/res.txt
#SBATCH --mail-user=ghonimaraghda@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_mem:80gb"

# Hugging Face & Transformers Cache
export HF_HOME=/ukp-storage-1/ghonema/.cache/.hf_cache
export HF_HUB_CACHE=/ukp-storage-1/ghonema/.cache/.hf_cache
export TRANSFORMERS_CACHE=/ukp-storage-1/ghonema/.cache/.hf_cache
export HF_HOME=/ukp-storage-1/ghonema/.cache/.hf_cache
# Datasets Cache
export HF_DATASETS_CACHE=/ukp-storage-1/ghonema/.cache/.hf_cache/datasets

# Temp Directory
export TMPDIR=/ukp-storage-1/ghonema/.cache/.tmp


# ----- Paths & caches -----
export PYTHONPATH="$WORKSPACE_ROOT:${PYTHONPATH:-}"
export HF_HOME=/ukp-storage-1/ghonema/.cache/.hf_cache
export HF_HUB_CACHE=/ukp-storage-1/ghonema/.cache/.hf_cache
export TRANSFORMERS_CACHE=/ukp-storage-1/ghonema/.cache/.hf_cache
export HF_DATASETS_CACHE=/ukp-storage-1/ghonema/.cache/.hf_cache/datasets
export TMPDIR=/ukp-storage-1/ghonema/.cache/.tmp
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$WORKSPACE_ROOT/.config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_ROOT/.cache}"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
export VLLM_DISABLE_USAGE_COLLECTION=1
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$XDG_CACHE_HOME/vllm}"
mkdir -p "$VLLM_CACHE_DIR"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$TRITON_CACHE_DIR"

# Hard override everything
export HOME=/ukp-storage-1/ghonema

# TorchDynamo / Inductor cache
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor
mkdir -p $TORCHINDUCTOR_CACHE_DIR

# Triton cache
export TRITON_CACHE_DIR=$HOME/.cache/triton
mkdir -p $TRITON_CACHE_DIR

# CUDA cache
export CUDA_CACHE_PATH=$HOME/.cache/cuda
mkdir -p $CUDA_CACHE_PATH

# Generic cache fallbacks
export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME

export VLLM_CACHE_ROOT=$HOME/.cache

source /ukp-storage-1/ghonema/miniconda3/bin/activate /ukp-storage-1/ghonema/miniconda3/envs/thesis310


srun python /ukp-storage-1/ghonema/agentic/main.py
