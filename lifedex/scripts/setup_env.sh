#!/usr/bin/env bash
# Create the `lifedex` conda env: a clone of the existing `openpi` env (left untouched, it is used by another
# project) + dexjoco's openpi fork, a LeRobot version that reads v3.0 datasets, and TensorBoard.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/mnt/data/yuanmingqi/miniconda3}"
SRC_ENV="${SRC_ENV:-openpi}"
ENV_NAME="${ENV_NAME:-lifedex}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA="$CONDA_ROOT/bin/conda"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

if [[ ! -x "$PY" ]]; then
  "$CONDA" create --clone "$SRC_ENV" -n "$ENV_NAME" -y
fi

# ffmpeg with AV1 (dav1d) decoding for the LeRobot videos; torchcodec links against it.
"$CONDA" install -n "$ENV_NAME" -y -c conda-forge "ffmpeg=7"

"$PY" -m pip install -e "$REPO/dexjoco/openpi"
"$PY" -m pip install -e "$REPO/dexjoco/openpi/packages/openpi-client"
# Same as dexjoco/openpi/install.bash: only the dataset interface of LeRobot is needed, deps conflict with openpi.
"$PY" -m pip install "lerobot==${LEROBOT_VERSION:-0.4.4}" --no-deps
"$PY" -m pip install tensorboard

"$PY" - <<'EOF'
import jax, lerobot, openpi, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
from torch.utils.tensorboard import SummaryWriter  # noqa: F401
print("openpi:", openpi.__file__)
print("jax", jax.__version__, jax.devices())
print("torch", torch.__version__, "lerobot", lerobot.__version__)
EOF
