"""Evaluate every finished stage of a LifeDex run in DexJoCo and summarize the success-rate matrix and NBT.

    python -m lifedex.eval.run_eval --run-dir checkpoints/lifedex/seq_ft                  # all finished stages
    python -m lifedex.eval.run_eval --run-dir ... --episodes 2 --seeds 0 --out-dir /tmp/x  # quick check

Protocol (DexJoCo Table II): per (checkpoint, task) 3 seeds x 50 episodes with the stock `dexjoco-openpi-eval` client
(asynchronous chunked inference, real-time 30 Hz simulation). By default each checkpoint is evaluated on the tasks
learned so far (lower-triangular matrix); `--scope all` also evaluates not-yet-learned tasks.

Work unit = (stage, task, seed). Units run in parallel on "slots" (GPUs x servers_per_gpu); a slot keeps one policy
server (lifedex.eval.serve) alive and reuses it for units of the same stage. Because the simulation runs in real time,
inference latency affects success rates: the servers log `INFER_STATS` (keep p95 well below the ~200 ms between
replans, e.g. by not packing too many servers per GPU). Finished units are skipped on rerun (resumable).

Outputs under <out_dir> (default <run_dir>/eval/<regime>):
    stage01_click_mouse/hammer_nail/seed0/{episode_xx_success|failure/, client.log, result.json}
    servers/*.log, results.json, summary.md
"""

import dataclasses
import json
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import tyro

from lifedex import tasks as _tasks
from lifedex.eval import metrics as _metrics
from lifedex.paths import REPO_ROOT

DEXJOCO_ROOT = REPO_ROOT / "dexjoco"
CONDA_ROOT = pathlib.Path(os.environ.get("CONDA_ROOT", "/mnt/data/yuanmingqi/miniconda3"))


@dataclasses.dataclass
class Args:
    run_dir: pathlib.Path
    # Stages (checkpoints) to evaluate; default: every finished stage.
    stages: tuple[int, ...] | None = None
    # "seen": tasks 0..k for checkpoint k (lower triangle); "all": every task of the stream.
    scope: str = "seen"
    # Restrict to these tasks (default: all tasks of the run).
    tasks: tuple[str, ...] | None = None
    # Evaluation regime; default: the run's first training regime.
    regime: _tasks.Regime | None = None
    seeds: tuple[int, ...] = (0, 1, 2)
    episodes: int = 50
    # GPUs for policy servers (default: all visible) and servers per GPU.
    gpus: tuple[int, ...] | None = None
    servers_per_gpu: int = 1
    # GPUs used by the simulator clients for MuJoCo EGL rendering (round-robin over slots). Default: the slot's server
    # GPU. Rendering on the inference GPU time-slices it (measured on H20: inference 95 ms alone -> 150 ms with one
    # rendering client, 250 ms with three), so on multi-GPU nodes dedicate a few GPUs to rendering, e.g.
    # `--gpus 0 1 2 3 4 5 --render-gpus 6 7`. CPU software rendering is ~15-20x slower than EGL and not supported.
    render_gpus: tuple[int, ...] | None = None
    base_port: int = 18000
    out_dir: pathlib.Path | None = None
    # Keep episode videos only for these seeds (others are deleted after the unit finishes to save space).
    video_seeds: tuple[int, ...] = (0,)
    # Retries per unit (server restarted in between).
    retries: int = 1
    server_python: str = sys.executable
    eval_bin: str = str(CONDA_ROOT / "envs" / "dexjoco" / "bin" / "dexjoco-openpi-eval")
    summarize_only: bool = False
    dry_run: bool = False


@dataclasses.dataclass(frozen=True, order=True)
class Unit:
    stage: int
    task: str
    seed: int


class RunInfo:
    def __init__(self, run_dir: pathlib.Path):
        self.run_dir = run_dir.resolve()
        c = json.loads((self.run_dir / "config.json").read_text())
        resolved = c.get("resolved", {})
        self.tasks: list[str] = list(resolved.get("tasks") or c.get("tasks") or _tasks.STREAMS[c["stream"]])
        self.layout: str = resolved.get("layout") or _tasks.default_layout(self.tasks)
        self.regimes: list[str] = list(c.get("regimes") or ["rand_obj"])
        self.exp_name: str = c["exp_name"]
        self.method: str = c["method"]

    def stage_dir(self, k: int) -> pathlib.Path:
        return self.run_dir / f"stage{k:02d}_{self.tasks[k]}"

    def finished_stages(self) -> list[int]:
        return [k for k in range(len(self.tasks)) if (self.stage_dir(k) / "DONE.json").exists()]


def eval_config(info: RunInfo, task: str, regime: str) -> tuple[pathlib.Path, list[str]]:
    """dexjoco eval YAML + extra client flags for a task under the run's layout."""
    spec = _tasks.TASKS[task]
    extra = ["--rand-full"] if regime == "rand_full" else []
    if info.layout == "single_arm":
        return DEXJOCO_ROOT / "configs" / regime / f"{task}.yaml", extra
    if regime == "rand_full":
        raise NotImplementedError("bimanual layout + rand_full: dexjoco has no multi_task rand_full eval configs.")
    return DEXJOCO_ROOT / "configs" / "multi_task" / f"{task}.yaml", extra + (["--pad-state-dim46"] if spec.single_arm else [])


def read_unit_result(unit_dir: pathlib.Path) -> dict | None:
    f = unit_dir / "result.json"
    return json.loads(f.read_text()) if f.exists() else None


def visible_gpus() -> list[int]:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()] or [0]


class Slot:
    """One policy server on one GPU, reused across units of the same stage."""

    def __init__(self, idx: int, gpu: int, port: int, mem_fraction: float, args: Args, info: RunInfo, log_dir: pathlib.Path,
                 render_gpu: int | None = None):
        self.idx, self.gpu, self.port = idx, gpu, port
        self.render_gpu = gpu if render_gpu is None else render_gpu
        self.mem_fraction = mem_fraction
        self.args, self.info, self.log_dir = args, info, log_dir
        self.proc: subprocess.Popen | None = None
        self.stage: int | None = None

    def _env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS",)}
        env.update(
            CUDA_VISIBLE_DEVICES=str(self.gpu),
            XLA_PYTHON_CLIENT_MEM_FRACTION=f"{self.mem_fraction:.3f}",
            PYTHONPATH=str(REPO_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            OPENPI_DATA_HOME=env.get("OPENPI_DATA_HOME", str(REPO_ROOT / ".cache" / "openpi")),
            JAX_COMPILATION_CACHE_DIR=env.get("JAX_COMPILATION_CACHE_DIR", str(REPO_ROOT / ".cache" / "jax")),
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
        )
        return env

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=2) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def ensure(self, stage: int):
        if self.stage == stage and self.proc is not None and self.proc.poll() is None:
            return
        self.stop()
        log = self.log_dir / f"slot{self.idx}_gpu{self.gpu}_stage{stage:02d}.log"
        cmd = [self.args.server_python, "-m", "lifedex.eval.serve", "--run-dir", str(self.info.run_dir),
               "--stage", str(stage), "--port", str(self.port)]
        logging.info(f"[slot {self.idx}] starting server for stage {stage} on GPU {self.gpu}:{self.port}")
        with open(log, "a") as f:
            self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=self._env(),
                                         start_new_session=True)
        t0 = time.time()
        while not self.healthy():
            if self.proc.poll() is not None:
                raise RuntimeError(f"[slot {self.idx}] server died (see {log}):\n" + _tail(log))
            if time.time() - t0 > 1800:
                raise TimeoutError(f"[slot {self.idx}] server not ready after 30 min (see {log})")
            time.sleep(3)
        self.stage = stage
        logging.info(f"[slot {self.idx}] server for stage {stage} ready after {time.time() - t0:.0f}s")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.proc, self.stage = None, None


def _tail(path: pathlib.Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return ""


def run_unit(unit: Unit, slot: Slot, args: Args, info: RunInfo, regime: str, unit_dir: pathlib.Path) -> dict:
    config, extra = eval_config(info, unit.task, regime)
    if unit_dir.exists():
        shutil.rmtree(unit_dir)  # partial results of an interrupted unit
    unit_dir.mkdir(parents=True)
    cmd = [args.eval_bin, "--config", str(config), "--seed", str(unit.seed), "--port", str(slot.port),
           "--host", "127.0.0.1", "--episodes", str(args.episodes), "--output", str(unit_dir), *extra]
    env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(slot.render_gpu), PYOPENGL_PLATFORM="egl")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    # Episodes last <= 40 s of real time; generous bound for resets / slow rendering.
    timeout = 300 + 120 * args.episodes
    t0 = time.time()
    with open(unit_dir / "client.log", "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=DEXJOCO_ROOT, env=env, timeout=timeout)
    marks = sorted(unit_dir.glob("success_rate_*_*.txt"))
    if proc.returncode != 0 or not marks:
        raise RuntimeError(f"client failed (rc={proc.returncode}) for {unit}:\n" + _tail(unit_dir / "client.log"))
    n_succ, n_eps = (int(x) for x in marks[-1].stem.split("_")[-2:])
    result = {"stage": unit.stage, "task": unit.task, "seed": unit.seed, "regime": regime, "success": n_succ,
              "episodes": n_eps, "success_rate": n_succ / n_eps, "wall_time_s": round(time.time() - t0, 1),
              "checkpoint_stage_task": info.tasks[unit.stage]}
    if unit.seed not in args.video_seeds:
        for mp4 in unit_dir.glob("episode_*/*.mp4"):
            mp4.unlink()
    (unit_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def collect(out_dir: pathlib.Path, info: RunInfo) -> dict[tuple[int, str, int], float]:
    results = {}
    for f in out_dir.glob("stage*/*/seed*/result.json"):
        r = json.loads(f.read_text())
        results[(r["stage"], r["task"], r["seed"])] = r["success_rate"]
    return results


def write_summary(out_dir: pathlib.Path, info: RunInfo, regime: str, seeds: list[int]) -> str:
    results = collect(out_dir, info)
    summary = _metrics.summarize(results, info.tasks, seeds)
    md = _metrics.to_markdown(summary, info.tasks, regime=regime, title=f"{info.exp_name} ({info.method})")
    (out_dir / "summary.md").write_text(md)
    (out_dir / "results.json").write_text(json.dumps({
        "run_dir": str(info.run_dir), "regime": regime, "tasks": info.tasks, "seeds": seeds,
        "units": [{"stage": s, "task": t, "seed": sd, "success_rate": v} for (s, t, sd), v in sorted(results.items())],
        "matrix_mean": summary["mean"].tolist(), "matrix_std": summary["std"].tolist(),
        "metrics": {k: {"mean": m, "std": s} for k, (m, s) in summary["metrics"].items()},
    }, indent=2, default=lambda x: None))
    return md


def main(args: Args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname).1s] %(message)s", force=True)
    info = RunInfo(args.run_dir)
    regime = args.regime or info.regimes[0]
    out_dir = (args.out_dir or info.run_dir / "eval" / regime).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)

    if args.summarize_only:
        print(write_summary(out_dir, info, regime, seeds))
        return
    if not pathlib.Path(args.eval_bin).exists():
        raise FileNotFoundError(f"{args.eval_bin} not found; create the dexjoco env (lifedex/scripts/setup_eval_env.sh).")

    finished = info.finished_stages()
    stages = [k for k in (args.stages if args.stages is not None else finished)]
    missing = [k for k in stages if k not in finished]
    if missing:
        raise ValueError(f"Stages {missing} are not finished yet (finished: {finished}).")
    eval_tasks = [t for t in info.tasks if args.tasks is None or t in args.tasks]
    units = sorted(
        Unit(k, t, s)
        for k in stages
        for t in eval_tasks
        if args.scope == "all" or info.tasks.index(t) <= k
        for s in seeds
    )
    unit_dir = lambda u: out_dir / f"stage{u.stage:02d}_{info.tasks[u.stage]}" / u.task / f"seed{u.seed}"  # noqa: E731
    todo = [u for u in units if read_unit_result(unit_dir(u)) is None]
    logging.info(f"Run {info.run_dir} | layout={info.layout} | regime={regime} | stages={stages} | "
                 f"{len(units)} units, {len(todo)} to run, {args.episodes} episodes each -> {out_dir}")
    if args.dry_run or not todo:
        for u in todo:
            logging.info(f"  todo: {u}")
        print(write_summary(out_dir, info, regime, seeds))
        return

    gpus = list(args.gpus) if args.gpus is not None else visible_gpus()
    n_slots = min(len(gpus) * args.servers_per_gpu, len(todo))
    log_dir = out_dir / "servers"
    log_dir.mkdir(exist_ok=True)
    render_gpus = list(args.render_gpus) if args.render_gpus else None
    if render_gpus and set(render_gpus) & set(gpus):
        logging.warning(f"--render-gpus {render_gpus} overlap with server GPUs {gpus}: inference will be slowed there.")
    slots = [Slot(i, gpus[i % len(gpus)], args.base_port + i, 0.85 / args.servers_per_gpu, args, info, log_dir,
                  render_gpu=render_gpus[i % len(render_gpus)] if render_gpus else None)
             for i in range(n_slots)]
    logging.info(f"{n_slots} slot(s): servers on GPUs {gpus} ({args.servers_per_gpu}/GPU), rendering on "
                 f"{render_gpus or 'the server GPU'}")

    lock = threading.Lock()
    queue = list(todo)
    failures: list[tuple[Unit, str]] = []
    done_count = [0]
    t_start = time.time()

    def next_unit(stage: int | None) -> Unit | None:
        with lock:
            if not queue:
                return None
            for i, u in enumerate(queue):  # prefer the stage whose server is already loaded
                if u.stage == stage:
                    return queue.pop(i)
            return queue.pop(0)

    def worker(slot: Slot):
        try:
            while (u := next_unit(slot.stage)) is not None:
                for attempt in range(args.retries + 1):
                    try:
                        slot.ensure(u.stage)
                        r = run_unit(u, slot, args, info, regime, unit_dir(u))
                        with lock:
                            done_count[0] += 1
                            el = time.time() - t_start
                            eta = el / done_count[0] * (len(todo) - done_count[0]) / 60
                            logging.info(f"[slot {slot.idx}] {u}: {r['success']}/{r['episodes']} "
                                         f"({r['wall_time_s']:.0f}s) | {done_count[0]}/{len(todo)} done, ETA {eta:.0f} min")
                        break
                    except Exception as e:  # noqa: BLE001
                        logging.error(f"[slot {slot.idx}] {u} attempt {attempt} failed: {e}")
                        slot.stop()
                        if attempt == args.retries:
                            with lock:
                                failures.append((u, str(e).splitlines()[0]))
        finally:
            slot.stop()

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in slots]
    try:
        for th in threads:
            th.start()
            time.sleep(5)  # stagger checkpoint loading
        for th in threads:
            th.join()
    finally:
        for s in slots:
            s.stop()

    md = write_summary(out_dir, info, regime, seeds)
    print(md)
    if failures:
        logging.error(f"{len(failures)} unit(s) failed: " + "; ".join(f"{u}: {e}" for u, e in failures))
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
