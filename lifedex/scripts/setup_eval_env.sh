#!/usr/bin/env bash
# Create the `dexjoco` conda env used by the evaluation client (MuJoCo 3.4 simulation + openpi websocket client).
# The policy servers run in the `lifedex` env (see setup_env.sh).
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/mnt/data/yuanmingqi/miniconda3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -x "$CONDA_ROOT/envs/dexjoco/bin/python" ]]; then
  (cd "$REPO/dexjoco" && "$CONDA_ROOT/bin/conda" env create -f environment-dexjoco.yaml)
fi
MUJOCO_GL=egl "$CONDA_ROOT/envs/dexjoco/bin/python" - <<'EOF'
import mujoco
from dexjoco.tasks import CONFIG_MAPPING
print("mujoco", mujoco.__version__, "| tasks:", sorted(CONFIG_MAPPING))
EOF
