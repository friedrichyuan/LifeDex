#!/usr/bin/env bash
# Evaluate all finished stages of a run (success-rate matrix + NBT), single node, all visible GPUs.
#
#   bash lifedex/scripts/dlc_eval.sh <run_dir> [extra run_eval flags...]
#   bash lifedex/scripts/dlc_eval.sh checkpoints/lifedex/seq_ft
#   bash lifedex/scripts/dlc_eval.sh checkpoints/lifedex/seq_ft --stages 0 1 --seeds 0 --episodes 10   # quick
#   bash lifedex/scripts/dlc_eval.sh checkpoints/lifedex/seq_ft --summarize-only
#
# Default protocol = DexJoCo Table II: 3 seeds x 50 episodes per (checkpoint, task), tasks learned so far.
# Resumable: finished (stage, task, seed) units are skipped. Results: <run_dir>/eval/<regime>/summary.md
set -euo pipefail

RUN_DIR="${1:?run_dir}"
shift

CONDA_ROOT="${CONDA_ROOT:-/mnt/data/yuanmingqi/miniconda3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$CONDA_ROOT/envs/${ENV_NAME:-lifedex}/bin/python"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO/.cache/openpi}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$REPO/.cache/jax}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CONDA_ROOT

cd "$REPO"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
exec "$PY" -m lifedex.eval.run_eval --run-dir "$RUN_DIR" "$@"
