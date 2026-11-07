"""Serve one stage checkpoint of a LifeDex run over openpi's websocket protocol (same protocol as dexjoco's
`openpi/scripts/serve_policy.py`, so the stock `dexjoco-openpi-eval` client works unchanged).

    python -m lifedex.eval.serve --run-dir checkpoints/lifedex/seq_ft --stage 1 --port 8000

Differences from dexjoco's server: the TrainConfig is rebuilt from the run's config.json (LifeDex configs are not
registered in openpi), and one warm-up inference is run *before* the port opens. DexJoCo simulates in real time, so
otherwise the JIT compilation of the first request would eat the first evaluation episode.
"""

import dataclasses
import json
import logging
import pathlib
import time

import numpy as np
import tyro

from lifedex import tasks as _tasks
from lifedex.config import load_run_config, make_train_config

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402


@dataclasses.dataclass
class Args:
    run_dir: pathlib.Path
    # Stage index whose final checkpoint is served (reads stage<k>_*/DONE.json).
    stage: int | None = None
    # Explicit checkpoint directory (<stage>/checkpoints/<step>); overrides --stage.
    checkpoint: str | None = None
    port: int = 8000
    host: str = "0.0.0.0"
    # Log inference-latency statistics every N requests (latency matters: DexJoCo runs in real time).
    stats_every: int = 300


class TimedPolicy:
    def __init__(self, policy, stats_every: int):
        self._policy = policy
        self._every = stats_every
        self._times: list[float] = []

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        t = time.perf_counter()
        out = self._policy.infer(obs)
        self._times.append((time.perf_counter() - t) * 1000)
        if len(self._times) >= self._every:
            ts = np.asarray(self._times)
            logging.info(
                f"INFER_STATS n={len(ts)} mean={ts.mean():.1f}ms p50={np.percentile(ts, 50):.1f}ms "
                f"p95={np.percentile(ts, 95):.1f}ms max={ts.max():.1f}ms"
            )
            self._times = []
        return out


def resolve_checkpoint(run_dir: pathlib.Path, stage: int) -> str:
    cfg = load_run_config(run_dir)
    done = cfg.stage_dir(stage) / "DONE.json"
    if not done.exists():
        raise FileNotFoundError(f"Stage {stage} of {run_dir} is not finished ({done} missing).")
    return json.loads(done.read_text())["checkpoint"]


def dummy_obs(layout: str) -> dict:
    img = np.zeros((224, 224, 3), np.uint8)
    state_dim = _tasks.LAYOUTS[layout][0]
    obs = {"state": np.zeros(state_dim, np.float32), "prompt": "warm up"}
    if layout == "single_arm":
        obs.update(base=img, wrist=img)
    else:
        obs.update(base=img, wrist_left=img, wrist_right=img)
    return obs


def main(args: Args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname).1s] %(message)s", force=True)
    cfg = load_run_config(args.run_dir)
    if args.checkpoint is None and args.stage is None:
        raise ValueError("Pass --stage or --checkpoint.")
    ckpt = args.checkpoint or resolve_checkpoint(args.run_dir, args.stage)
    # Any task of the run works: inference only depends on the run's layout, transforms and shared norm stats.
    train_config = make_train_config(cfg, _tasks.TASKS[cfg.task_names[0]], init_params="unused")
    logging.info(f"Loading {ckpt} (layout={cfg.resolved_layout}, asset_id={cfg.asset_id})")
    policy = _policy_config.create_trained_policy(train_config, ckpt)

    for i in range(3):  # compile + a couple of timed calls
        t = time.perf_counter()
        policy.infer(dummy_obs(cfg.resolved_layout))
        logging.info(f"Warm-up inference {i}: {(time.perf_counter() - t) * 1000:.0f} ms")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=TimedPolicy(policy, args.stats_every), host=args.host, port=args.port, metadata=policy.metadata
    )
    logging.info(f"SERVER_READY port={args.port} checkpoint={ckpt}")
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
