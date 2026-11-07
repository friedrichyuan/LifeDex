"""Repository paths. Everything is resolved to absolute paths so that jobs can run from any cwd (e.g. on DLC)."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIFEDEX_ROOT = REPO_ROOT / "lifedex"
OPENPI_ROOT = REPO_ROOT / "dexjoco" / "openpi"

DATA_ROOT = Path(os.environ.get("LIFEDEX_DATA_ROOT", REPO_ROOT / "data"))
MODELS_ROOT = DATA_ROOT / "dexjoco_models"
# pi0.5 base checkpoint (32-dim action head); used for the native single-arm layout.
BASE_PARAMS = MODELS_ROOT / "pi05_base" / "params"
# pi0.5 base checkpoint with the action projections expanded to 44 dims (bimanual / unified layout).
BASE_PARAMS_44 = MODELS_ROOT / "pi05_base_action_dim_44" / "params"

# Shared normalization statistics, one sub-directory per task stream (asset id).
ASSETS_ROOT = LIFEDEX_ROOT / "assets"
CKPT_ROOT = Path(os.environ.get("LIFEDEX_CKPT_ROOT", REPO_ROOT / "checkpoints" / "lifedex"))
