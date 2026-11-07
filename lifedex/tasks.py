"""DexJoCo task registry, data regimes and task streams.

Two layouts:

* ``single_arm`` (native, = dexjoco single-task SFT): pi05_base (32-dim action head), 22-dim action, 23-dim state,
  2 cameras (third-person -> base slot, wrist -> left-wrist slot, right-wrist slot masked out). Only single-arm tasks.
  Evaluated with dexjoco/configs/{rand_obj,rand_full}/<task>.yaml.
* ``bimanual`` (unified, = dexjoco ``multi_task`` convention): pi05_base_action_dim_44, state/action zero-padded to
  46/44, single-arm wrist image fed to both wrist slots. Needed only when a stream mixes single-arm and bimanual tasks.
  Evaluated with dexjoco/configs/multi_task/<task>.yaml and ``--pad-state-dim46``.

Two data regimes (separate LeRobot datasets with identical task names):

* ``rand_obj``: object placement + table height randomization.
* ``rand_full``: rand_obj + third-person camera pose, lighting and table texture randomization; the third-person view
  is stored as ``observation.images.random_camera``.
"""

import dataclasses
from pathlib import Path
from typing import Literal

from lifedex.paths import BASE_PARAMS, BASE_PARAMS_44, DATA_ROOT

Layout = Literal["single_arm", "bimanual"]
Regime = Literal["rand_obj", "rand_full"]

REGIME_ROOTS: dict[str, Path] = {
    "rand_obj": DATA_ROOT / "dexjoco_datasets" / "dexjoco_lerobot_datasets",
    "rand_full": DATA_ROOT / "dexjoco_datasets" / "dexjoco_lerobot_datasets_rand_full",
}

# (state_dim, action_dim, model action_dim, initial weights)
LAYOUTS: dict[str, tuple[int, int, int, Path]] = {
    "single_arm": (23, 22, 32, BASE_PARAMS),
    "bimanual": (46, 44, 44, BASE_PARAMS_44),
}


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    name: str
    single_arm: bool
    # Camera keys in the rand_obj dataset; rand_full replaces the third-person view with `random_camera`.
    base_img: str
    wrist_left_img: str
    wrist_right_img: str

    def data_root(self, regime: Regime) -> Path:
        return REGIME_ROOTS[regime] / self.name

    def base_img_for(self, regime: Regime) -> str:
        return "observation.images.random_camera" if regime == "rand_full" else self.base_img


def _single(name: str, base_img: str = "observation.images.front") -> TaskSpec:
    wrist = "observation.images.wrist"
    return TaskSpec(name=name, single_arm=True, base_img=base_img, wrist_left_img=wrist, wrist_right_img=wrist)


def _bimanual(name: str) -> TaskSpec:
    return TaskSpec(
        name=name,
        single_arm=False,
        base_img="observation.images.ego",
        wrist_left_img="observation.images.wrist_left",
        wrist_right_img="observation.images.wrist_right",
    )


TASKS: dict[str, TaskSpec] = {
    t.name: t
    for t in [
        _single("hammer_nail"),
        _single("click_mouse", base_img="observation.images.ego_right"),
        _single("pick_bucket"),
        _single("water_plant"),
        _single("fold_glasses"),
        _single("pinch_tongs"),
        _bimanual("bimanual_photograph"),
        _bimanual("bimanual_assembly"),
        _bimanual("bimanual_hanoi"),
        _bimanual("bimanual_microwave_cook"),
        _bimanual("bimanual_unlock_ipad"),
    ]
}

# name -> ordered task list.
STREAMS: dict[str, tuple[str, ...]] = {
    # Five single-arm tasks with high single-task SFT success rates.
    "dex5": ("hammer_nail", "click_mouse", "pick_bucket", "water_plant", "fold_glasses"),
    # Previous stream mixing in a bimanual task (requires layout="bimanual").
    "dex5_bi": ("hammer_nail", "click_mouse", "pick_bucket", "water_plant", "bimanual_photograph"),
}


def get_tasks(names) -> list[TaskSpec]:
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Available: {sorted(TASKS)}")
    return [TASKS[n] for n in names]


def default_layout(names) -> Layout:
    return "single_arm" if all(t.single_arm for t in get_tasks(names)) else "bimanual"


def check_layout(names, layout: Layout) -> None:
    if layout == "single_arm" and not all(t.single_arm for t in get_tasks(names)):
        raise ValueError(f"layout='single_arm' cannot hold bimanual tasks: {list(names)}. Use layout='bimanual'.")


def asset_id(stream: str, layout: Layout, regimes: tuple[str, ...]) -> str:
    """Directory name of the shared norm stats for a (stream, layout, data regimes) combination."""
    return f"{stream}_{layout}_{'+'.join(regimes)}"
