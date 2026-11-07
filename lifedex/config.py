"""Continual-learning run configuration and the openpi configs derived from it."""

import dataclasses
import pathlib
from typing import Literal

import tyro

from lifedex import tasks as _tasks
from lifedex.openpi_bootstrap import import_openpi_config
from lifedex.paths import ASSETS_ROOT, CKPT_ROOT

_config = import_openpi_config()

import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402

Method = Literal["seq_ft", "er", "ewc"]


@dataclasses.dataclass(frozen=True)
class ERConfig:
    # Replay samples (frame + action chunk) stored per previous task. Values in (0, 1) are a fraction of the task's
    # frames instead. LIBERO/the reference paper use 1000 per task (~20% of a LIBERO task; ~2-5% of a DexJoCo task).
    buffer_size: float = 1000
    # Fraction of every batch drawn from the replay buffer (0.5 = 1:1 current vs. all past tasks, as in LIBERO).
    replay_ratio: float = 0.5


@dataclasses.dataclass(frozen=True)
class EWCConfig:
    # Penalty weight: loss + lam / 2 * sum_i F_i (theta_i - theta*_i)^2, with F normalized to unit mean (see below).
    # With Adam, a parameter of average Fisher settles where lam * F * |d| ~ |task grad| ~ sqrt(F_raw), i.e.
    # |d| ~ 1e-4 / lam for pi0.5 on DexJoCo (raw Fisher mean ~1e-8). lam=1 nearly freezes the model (smoke test:
    # penalty grad ~35x task grad after 2 steps); sweep e.g. {1e-3, 1e-2, 1e-1, 1}.
    lam: float = 1e-2
    # Online EWC decay: F <- gamma * F_prev + F_task. gamma=1 sums Fishers of all past tasks.
    gamma: float = 1.0
    # Number of mini-batches used to estimate the diagonal (empirical) Fisher after each task.
    fisher_batches: int = 100
    # Batch size for Fisher estimation (smaller = closer to the per-sample Fisher). None -> train batch size.
    fisher_batch_size: int | None = None
    # Rescale each task's Fisher to unit mean before accumulation so that `lam` is comparable across tasks/models.
    normalize_fisher: bool = True


@dataclasses.dataclass(frozen=True)
class CLConfig:
    # Run name; checkpoints/logs go to <ckpt_root>/<exp_name>.
    exp_name: str = tyro.MISSING
    method: Method = "seq_ft"
    # Task stream (see lifedex/tasks.py:STREAMS).
    stream: str = "dex5"
    # Optional explicit task list overriding the stream's tasks.
    tasks: tuple[str, ...] | None = None
    # State/action/camera layout; None -> "single_arm" if all tasks are single-arm, else "bimanual".
    layout: _tasks.Layout | None = None
    # Training data regime(s). Several regimes are concatenated per task (each task = union of its regime datasets).
    regimes: tuple[_tasks.Regime, ...] = ("rand_obj",)

    # Optimization (defaults reproduce dexjoco's single-task pi0.5 LoRA SFT recipe, applied per task).
    steps_per_task: int = 30_000
    batch_size: int = 32
    peak_lr: float = 5e-5
    warmup_steps: int = 10_000
    decay_steps: int = 1_000_000
    decay_lr: float = 5e-5
    clip_gradient_norm: float = 1.0
    seed: int = 42
    # Data-loader worker processes. AV1 video decoding is the throughput bottleneck on fast GPUs (~15-20 samples/s per
    # worker); set close to the number of CPU cores of the node.
    num_workers: int = 16
    fsdp_devices: int = 1

    log_interval: int = 100
    save_interval: int = 5_000

    # Initial weights for the first task; None -> the layout's base model (pi05_base / pi05_base_action_dim_44).
    init_params: str | None = None
    ckpt_root: pathlib.Path = CKPT_ROOT
    assets_root: pathlib.Path = ASSETS_ROOT

    er: ERConfig = dataclasses.field(default_factory=ERConfig)
    ewc: EWCConfig = dataclasses.field(default_factory=EWCConfig)

    @property
    def task_names(self) -> tuple[str, ...]:
        return self.tasks if self.tasks is not None else _tasks.STREAMS[self.stream]

    @property
    def resolved_layout(self) -> _tasks.Layout:
        return self.layout or _tasks.default_layout(self.task_names)

    @property
    def asset_id(self) -> str:
        return _tasks.asset_id(self.stream, self.resolved_layout, self.regimes)

    @property
    def base_params(self) -> str:
        return self.init_params or str(_tasks.LAYOUTS[self.resolved_layout][3])

    @property
    def run_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.ckpt_root) / self.exp_name).resolve()

    def stage_dir(self, k: int) -> pathlib.Path:
        return self.run_dir / f"stage{k:02d}_{self.task_names[k]}"


def model_config(layout: _tasks.Layout) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=_tasks.LAYOUTS[layout][2],
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        max_token_len=250,
    )


def make_train_config(
    cfg: CLConfig,
    task: _tasks.TaskSpec,
    *,
    init_params: str,
    regime: _tasks.Regime | None = None,
    batch_size: int | None = None,
):
    """openpi TrainConfig for training one stage on `task` (data of one `regime`).

    Also usable for serving any stage checkpoint of the run: at inference only the data/model transforms and norm stats
    matter, which are shared by all tasks and regimes of a run.
    """
    from lifedex.data import UnifiedDexDataConfig  # noqa: PLC0415  (avoid import cycle)

    regime = regime or cfg.regimes[0]
    layout = cfg.resolved_layout
    model = model_config(layout)
    assets = _config.AssetsConfig(assets_dir=str(cfg.assets_root), asset_id=cfg.asset_id)
    base = _config.DataConfig(prompt_from_task=True)
    if layout == "single_arm":
        if not task.single_arm:
            raise ValueError(f"Task {task.name} is bimanual; it needs layout='bimanual'.")
        data = _config.SingleArmDataConfig(
            repo_id="local_repo",
            root=task.data_root(regime),
            base_img_name=task.base_img_for(regime),
            assets=assets,
            base_config=base,
        )
    else:
        data = UnifiedDexDataConfig(
            repo_id="local_repo",
            root=task.data_root(regime),
            task=task,
            base_img=task.base_img_for(regime),
            assets=assets,
            base_config=base,
        )
    return _config.TrainConfig(
        name=f"lifedex_{cfg.asset_id}",
        exp_name=cfg.exp_name,
        model=model,
        data=data,
        batch_size=batch_size or cfg.batch_size,
        num_workers=cfg.num_workers,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=cfg.warmup_steps,
            peak_lr=cfg.peak_lr,
            decay_steps=cfg.decay_steps,
            decay_lr=cfg.decay_lr,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=cfg.clip_gradient_norm),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(init_params),
        num_train_steps=cfg.steps_per_task,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        keep_period=None,
        seed=cfg.seed,
        wandb_enabled=False,
        fsdp_devices=cfg.fsdp_devices,
        assets_base_dir=str(cfg.assets_root),
        checkpoint_base_dir=str(cfg.ckpt_root),
    )


PRESETS: dict[str, tuple[str, CLConfig]] = {
    "seq_ft": ("Sequential fine-tuning (no forgetting mitigation).", CLConfig(method="seq_ft")),
    "er": ("Experience replay with a fixed per-task buffer.", CLConfig(method="er")),
    "ewc": ("Online elastic weight consolidation.", CLConfig(method="ewc")),
}


def cli() -> CLConfig:
    return tyro.extras.overridable_config_cli(PRESETS)


def load_run_config(run_dir: str | pathlib.Path) -> CLConfig:
    """Rebuilds the CLConfig of a training run from <run_dir>/config.json (written by train_cl)."""
    import json  # noqa: PLC0415

    d = json.loads((pathlib.Path(run_dir) / "config.json").read_text())
    d.pop("resolved", None)
    fields = {f.name for f in dataclasses.fields(CLConfig)}
    d = {k: v for k, v in d.items() if k in fields}
    er = ERConfig(**d.pop("er", {}))
    ewc = EWCConfig(**d.pop("ewc", {}))
    for key in ("tasks", "regimes"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    for key in ("ckpt_root", "assets_root"):
        if d.get(key) is not None:
            d[key] = pathlib.Path(d[key])
    return CLConfig(**d, er=er, ewc=ewc)
