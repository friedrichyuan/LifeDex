#!/usr/bin/env bash
# Continual-learning training entry point for DLC (single node, all visible GPUs; data-parallel).
#
#   bash lifedex/scripts/dlc_train.sh <preset> <exp_name> [extra tyro overrides...]
#
# Presets: seq_ft | er | ewc. Examples:
#   bash lifedex/scripts/dlc_train.sh seq_ft seq_ft
#   bash lifedex/scripts/dlc_train.sh er er_m1000 --er.buffer-size 1000
#   bash lifedex/scripts/dlc_train.sh ewc ewc_lam1e-2 --ewc.lam 1e-2
#
# Shared norm stats are computed on the first run if missing (lifedex/assets/<stream>_<layout>_<regimes>).
# Data-loader workers default to (CPU cores - 4), capped at 64; AV1 decoding is the bottleneck on fast GPUs.
# Override with NUM_WORKERS=<n> or --num-workers <n>.
# The job is restartable: rerunning the same command resumes from the last finished stage / saved step.
set -euo pipefail

PRESET="${1:?preset (seq_ft|er|ewc)}"
EXP_NAME="${2:?exp_name}"
shift 2

CONDA_ROOT="${CONDA_ROOT:-/mnt/data/yuanmingqi/miniconda3}"
ENV_NAME="${ENV_NAME:-lifedex}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
export PATH="$CONDA_ROOT/envs/$ENV_NAME/bin:$PATH"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$REPO/.cache/jax}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# openpi downloads the PaliGemma tokenizer into this cache on first use.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO/.cache/openpi}"
# export LIFEDEX_CKPT_ROOT=/path/to/checkpoints   # default: $REPO/checkpoints/lifedex

CORES="$(nproc)"
DEFAULT_WORKERS=$(( CORES > 8 ? CORES - 4 : CORES ))
NUM_WORKERS="${NUM_WORKERS:-$(( DEFAULT_WORKERS > 64 ? 64 : DEFAULT_WORKERS ))}"

cd "$REPO"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo "CPU cores: $CORES, data-loader workers: $NUM_WORKERS"
# Later flags override earlier ones, so --num-workers in "$@" wins.
exec "$PY" -m lifedex.train_cl "$PRESET" --exp-name "$EXP_NAME" --num-workers "$NUM_WORKERS" "$@"
