"""Shared normalization statistics for a task stream.

All tasks (and data regimes) of a run share one set of stats so that a single policy serves every task. Stats are
computed from the LeRobot parquet files (no video decoding) over every frame, in the run's layout:
single_arm -> 23-dim state / 22-dim action; bimanual -> zero-padded to 46 / 44.

    python -m lifedex.compute_norm_stats --stream dex5                              # single_arm, rand_obj
    python -m lifedex.compute_norm_stats --stream dex5 --regimes rand_obj rand_full
Output: lifedex/assets/<stream>_<layout>_<regimes>/norm_stats.json (see lifedex/tasks.py:asset_id).
"""

import dataclasses
import pathlib

import numpy as np
import pandas as pd
import tqdm
import tyro

from lifedex import tasks as _tasks
from lifedex.paths import ASSETS_ROOT

import openpi.shared.normalize as normalize  # noqa: E402


@dataclasses.dataclass
class Args:
    stream: str = "dex5"
    tasks: tuple[str, ...] | None = None
    layout: _tasks.Layout | None = None
    regimes: tuple[_tasks.Regime, ...] = ("rand_obj",)
    assets_root: pathlib.Path = ASSETS_ROOT


def _load(root: pathlib.Path, state_dim: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "action"]) for f in files], ignore_index=True)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    pad = lambda x, d: np.pad(x, [(0, 0), (0, d - x.shape[1])])  # noqa: E731
    return pad(state, state_dim), pad(action, action_dim)


def main(args: Args):
    names = args.tasks if args.tasks is not None else _tasks.STREAMS[args.stream]
    layout = args.layout or _tasks.default_layout(names)
    _tasks.check_layout(names, layout)
    state_dim, action_dim, _, _ = _tasks.LAYOUTS[layout]

    states, actions = [], []
    for task in tqdm.tqdm(_tasks.get_tasks(names), desc="tasks"):
        for regime in args.regimes:
            s, a = _load(task.data_root(regime), state_dim, action_dim)
            print(f"{task.name} [{regime}]: {len(s)} frames")
            states.append(s)
            actions.append(a)
    # A single update: RunningStats fixes its quantile histogram range on the first batch and re-bins (lossily)
    # whenever later batches extend it, so feed everything at once.
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    stats["state"].update(np.concatenate(states))
    stats["actions"].update(np.concatenate(actions))
    norm_stats = {k: v.get_statistics() for k, v in stats.items()}

    out = args.assets_root / _tasks.asset_id(args.stream, layout, args.regimes)
    normalize.save(out, norm_stats)
    print(f"Wrote {out / 'norm_stats.json'} (state {state_dim}, action {action_dim})")


if __name__ == "__main__":
    main(tyro.cli(Args))
