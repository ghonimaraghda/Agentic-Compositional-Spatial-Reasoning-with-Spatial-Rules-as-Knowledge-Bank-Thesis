#!/bin/bash
#SBATCH --job-name=agentic_run
#SBATCH --output=/storage/ukp/work/ghonema/agentic/res.txt
#SBATCH --mail-user=ghonimaraghda@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_mem:80gb"
#SBATCH --mem=64G

#SBATCH --exclude=melvin,scratchy,penelope


set -euo pipefail

export WORKSPACE_ROOT=/storage/ukp/work/ghonema/agentic
export HOME=/storage/ukp/work/ghonema

# caches
export HF_HOME=$HOME/.cache/.hf_cache
export HF_HUB_CACHE=$HF_HOME
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE=$HF_HOME/datasets

export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_DATASETS_CACHE"

export VLLM_USE_V1=0
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONPATH="$WORKSPACE_ROOT:${PYTHONPATH:-}"

#Qwen2.5 +llama Env

ENV_PY=/storage/ukp/work/ghonema/miniconda3/envs/thesis310/bin/python


# QWEN3.5 env
# ENV_PY=/storage/ukp/work/ghonema/miniconda3/envs/qwen35/bin/python


# ============================================================
# Choose one run mode.
# Keep only one srun block uncommented.
# ============================================================

# ------------------------------------------------------------
# 1) Agentic compositional run
# ------------------------------------------------------------
srun $ENV_PY $WORKSPACE_ROOT/SpaRC_SpaRP_Spartun_P1.py \
 --model_path /storage/ukp/shared/shared_model_weights/models--Qwen2.5-72B-Instruct \
  --tp 2 \
  --max_tokens 768 \
  --outdir $WORKSPACE_ROOT/results \
  --log_every 20 

  

# ------------------------------------------------------------
# 2) Agentic directional frame-of-reference run
#check output fule in file outdir

# ------------------------------------------------------------
# srun $ENV_PY $WORKSPACE_ROOT/FoREST_P1.py \
#   --model_path /storage/ukp/shared/shared_model_weights/models--Qwen--Qwen3.5-27B \
#   --dataset_path /storage/ukp/work/ghonema/agentic/C-split_QA_camera_total.json \
#   --tp 2 \
#   --max_tokens 1024 \
#   --max_model_len 4096 \
#   --gpu_mem 0.95 \
#   --batch_size 32 \
#   --outdir $WORKSPACE_ROOT/results \
#   --log_every 20 



