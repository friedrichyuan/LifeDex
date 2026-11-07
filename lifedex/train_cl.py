"""Continual learning of pi0.5 on a stream of DexJoCo tasks: sequential fine-tuning, experience replay, online EWC.

Usage (see lifedex/scripts/*.sh):
    python -m lifedex.train_cl er --exp-name er_m1000 --er.buffer-size 1000

Layout of <ckpt_root>/<exp_name>/:
    config.json
    tb/                                   TensorBoard logs (global step across all stages)
    stage{k}_{task}/checkpoints/<step>/   openpi checkpoint (params/ is servable with the stage's TrainConfig)
    stage{k}_{task}/replay_manifest.json  ER: frame indices kept per previous task
    stage{k}_{task}/ewc_fisher.npz        EWC: accumulated Fisher after this stage
    stage{k}_{task}/DONE.json             written when the stage is complete (enables job-level resume)
"""

import dataclasses
import functools
import gc
import json
import logging
import os
import pathlib
import platform
import time

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter
import tqdm_loggable.auto as tqdm

from lifedex import data as _data
from lifedex import ewc as _ewc
from lifedex import tasks as _tasks
from lifedex.config import CLConfig, cli

import openpi.shared.array_typing as at  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.sharding as sharding  # noqa: E402
import openpi.training.utils as training_utils  # noqa: E402


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
        force=True,
    )


# --------------------------------------------------------------------------------------------------------------------
# Train state (mirrors openpi/scripts/train.py, which is a script and cannot be imported)
# --------------------------------------------------------------------------------------------------------------------
def _load_weights_and_validate(loader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def init_train_state(config, init_rng, mesh):
    """Fresh optimizer state + weights from `config.weight_loader` (previous stage or the base model)."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(init, donate_argnums=(1,), in_shardings=replicated, out_shardings=state_sharding)(
        init_rng, partial_params
    )
    return train_state, state_sharding


# --------------------------------------------------------------------------------------------------------------------
# Train step
# --------------------------------------------------------------------------------------------------------------------
def train_step(config, ewc_lam, rng, state: training_utils.TrainState, batch, is_replay, ewc_state):
    """One optimizer step. `ewc_state` is None or (fisher, anchor) over the trainable parameters."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        chunked = model.compute_loss(rng, observation, actions, train=True)
        per_sample = jnp.mean(chunked, axis=tuple(range(1, chunked.ndim)))
        return jnp.mean(per_sample), per_sample

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, per_sample), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )
    params = state.params.filter(config.trainable_filter)

    replay_w = is_replay.astype(jnp.float32)
    info = {
        "loss": loss,
        "loss_current": jnp.sum(per_sample * (1 - replay_w)) / jnp.maximum(jnp.sum(1 - replay_w), 1.0),
        "loss_replay": jnp.sum(per_sample * replay_w) / jnp.maximum(jnp.sum(replay_w), 1.0),
        "replay_fraction": jnp.mean(replay_w),
        "task_grad_norm": optax.global_norm(grads),
    }
    if ewc_state is not None:
        fisher, anchor = ewc_state
        penalty, pen_grad = _ewc.penalty_and_grad(params, fisher, anchor, ewc_lam)
        grads = jax.tree.map(jnp.add, grads, pen_grad)
        info["ewc_penalty"] = penalty
        info["ewc_grad_norm"] = optax.global_norm(pen_grad)
        info["total_loss"] = loss + 0.5 * ewc_lam * penalty

    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state)

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info["grad_norm"] = optax.global_norm(grads)
    info["param_norm"] = optax.global_norm(kernel_params)
    return new_state, info


# --------------------------------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------------------------------
def _done_file(cfg: CLConfig, k: int) -> epath.Path:
    return epath.Path(cfg.stage_dir(k)) / "DONE.json"


def _read_done(cfg: CLConfig, k: int) -> dict | None:
    f = _done_file(cfg, k)
    return json.loads(f.read_text()) if f.exists() else None


def _log_images(writer: SummaryWriter, batch, step: int, tag: str, n: int = 4):
    images = {k: np.asarray(jax.device_get(v)) for k, v in batch[0].images.items()}
    n = min(n, len(next(iter(images.values()))))
    grid = np.stack([np.concatenate([img[i] for img in images.values()], axis=1) for i in range(n)])
    if grid.dtype != np.uint8:  # openpi images are float in [-1, 1]
        grid = ((grid + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    writer.add_images(tag, grid, step, dataformats="NHWC")


def run_stage(cfg: CLConfig, k: int, task_specs, mesh, writer: SummaryWriter) -> dict:
    task = task_specs[k]
    stage_dir = cfg.stage_dir(k)
    stage_dir.mkdir(parents=True, exist_ok=True)
    init_params = cfg.base_params if k == 0 else _read_done(cfg, k - 1)["params"]
    global_offset = k * cfg.steps_per_task
    logging.info(
        f"===== Stage {k}: {task.name} (method={cfg.method}, layout={cfg.resolved_layout}, regimes={cfg.regimes}, "
        f"init={init_params}) ====="
    )

    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # ---- data
    current_ds, data_config, train_config = _data.create_task_dataset(cfg, task, init_params=init_params)
    replay = []
    if cfg.method == "er" and k > 0:
        manifest = {}
        for j in range(k):
            prev_ds, _, _ = _data.create_task_dataset(cfg, task_specs[j], init_params=init_params)
            idx = _data.replay_indices(len(prev_ds), cfg.er.buffer_size, seed=cfg.seed * 1000 + j)
            replay.append((prev_ds, idx))
            manifest[task_specs[j].name] = idx
        _data.save_replay_manifest(stage_dir / "replay_manifest.json", manifest)
    loader = _data.build_loader(
        current_ds,
        data_config,
        replay=replay,
        batch_size=cfg.batch_size,
        replay_ratio=cfg.er.replay_ratio,
        seed=cfg.seed + k,
        sharding=data_sharding,
        num_workers=cfg.num_workers,
    )
    data_iter = iter(loader)
    obs, actions, is_replay = next(data_iter)
    logging.info(f"Initialized data loader ({len(current_ds)} frames):\n{training_utils.array_tree_to_info((obs, actions))}")

    # ---- model / optimizer
    rng = jax.random.fold_in(jax.random.key(cfg.seed), k)
    train_rng, init_rng = jax.random.split(rng)
    train_state, state_sharding = init_train_state(train_config, init_rng, mesh)
    jax.block_until_ready(train_state)

    ewc_state = ewc_sharding = None
    if cfg.method == "ewc" and k > 0:
        trainable_sharding = state_sharding.params.filter(train_config.trainable_filter)
        params = train_state.params.filter(train_config.trainable_filter)
        anchor = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32, copy=True), params)
        fisher = _ewc.load(cfg.stage_dir(k - 1) / "ewc_fisher.npz", params)
        ewc_state = jax.device_put((fisher, anchor), (trainable_sharding, trainable_sharding))
        ewc_sharding = (trainable_sharding, trainable_sharding)
        del params

    mngr, resuming = _checkpoints.initialize_checkpoint_dir(
        stage_dir / "checkpoints", keep_period=None, overwrite=False, resume=True
    )
    if resuming:
        train_state = _checkpoints.restore_state(mngr, train_state, loader)
        logging.info(f"Resumed stage {k} from step {int(train_state.step)}")

    step_fn = functools.partial(train_step, train_config, cfg.ewc.lam)
    if ewc_state is None:
        ptrain_step = jax.jit(
            lambda r, s, b, m: step_fn(r, s, b, m, None),
            in_shardings=(replicated, state_sharding, data_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            step_fn,
            in_shardings=(replicated, state_sharding, data_sharding, data_sharding, ewc_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
    lr_fn = train_config.lr_schedule.create()

    start_step = int(train_state.step)
    num_steps = cfg.steps_per_task
    if start_step == 0:
        _log_images(writer, (obs, actions), global_offset, f"camera_views/stage{k:02d}_{task.name}")
        writer.add_text("stage", f"stage {k}: {task.name} (frames={len(current_ds)})", global_offset)

    pbar = tqdm.tqdm(range(start_step, num_steps), initial=start_step, total=num_steps, dynamic_ncols=True)
    infos, t0 = [], time.time()
    for step in pbar:
        with sharding.set_mesh(mesh):
            if ewc_state is None:
                train_state, info = ptrain_step(train_rng, train_state, (obs, actions), is_replay)
            else:
                train_state, info = ptrain_step(train_rng, train_state, (obs, actions), is_replay, ewc_state)
        infos.append(info)
        if step % cfg.log_interval == 0 or step == num_steps - 1:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            elapsed = time.time() - t0
            reduced["steps_per_sec"] = len(infos) / max(elapsed, 1e-6)
            reduced["lr"] = float(lr_fn(step))
            pbar.write(f"[stage {k} {task.name}] step {step}: " + ", ".join(f"{k_}={v:.4g}" for k_, v in reduced.items()))
            for key, val in reduced.items():
                writer.add_scalar(f"train/{key}", float(val), global_offset + step)
            writer.add_scalar("train/stage", k, global_offset + step)
            infos, t0 = [], time.time()
        obs, actions, is_replay = next(data_iter)
        if (step % cfg.save_interval == 0 and step > start_step) or step == num_steps - 1:
            _checkpoints.save_state(mngr, train_state, loader, step)
    mngr.wait_until_finished()
    final_step = mngr.latest_step()
    assert final_step == num_steps - 1, f"Expected final checkpoint at step {num_steps - 1}, found {final_step}."
    del data_iter, loader
    gc.collect()

    # ---- EWC: Fisher of this task, accumulated with previous tasks' (not needed after the last task)
    if cfg.method == "ewc" and k < len(task_specs) - 1:
        fisher_path = stage_dir / "ewc_fisher.npz"
        fisher_bs = cfg.ewc.fisher_batch_size or cfg.batch_size
        fisher_loader = _data.build_loader(
            current_ds,
            data_config,
            batch_size=fisher_bs,
            replay_ratio=0.0,
            seed=cfg.seed + 10_000 + k,
            sharding=data_sharding,
            num_workers=cfg.num_workers,
            num_batches=cfg.ewc.fisher_batches,
        )
        fisher_step = jax.jit(
            _ewc.make_fisher_step(train_config),
            in_shardings=(replicated, state_sharding, data_sharding),
        )
        logging.info(f"Estimating Fisher on {cfg.ewc.fisher_batches} batches of {fisher_bs} ...")
        with sharding.set_mesh(mesh):
            fisher_k = _ewc.estimate_fisher(
                fisher_step, jax.random.fold_in(rng, 7), train_state, fisher_loader, cfg.ewc.fisher_batches
            )
        writer.add_scalar("ewc/fisher_mean_raw", float(_ewc.tree_mean(fisher_k)), global_offset + num_steps - 1)
        if cfg.ewc.normalize_fisher:
            fisher_k = _ewc.normalize(fisher_k)
        if ewc_state is not None:
            fisher_k = jax.tree.map(lambda prev, cur: cfg.ewc.gamma * prev + cur, ewc_state[0], fisher_k)
        _ewc.save(fisher_path, fisher_k)
        del fisher_k, fisher_loader
        logging.info(f"Saved accumulated Fisher to {fisher_path}")

    done = {
        "task": task.name,
        "stage": k,
        "final_step": final_step,
        "params": str(stage_dir / "checkpoints" / str(final_step) / "params"),
        "checkpoint": str(stage_dir / "checkpoints" / str(final_step)),
    }
    _done_file(cfg, k).write_text(json.dumps(done, indent=2))
    del train_state, ewc_state
    gc.collect()
    return done


def main(cfg: CLConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()} with {jax.device_count()} device(s)")
    if cfg.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {cfg.batch_size} must be divisible by the number of devices {jax.device_count()}.")
    if cfg.method == "ewc" and (cfg.ewc.fisher_batch_size or cfg.batch_size) % jax.device_count() != 0:
        raise ValueError(f"EWC fisher batch size must be divisible by the number of devices {jax.device_count()}.")
    jax.config.update(
        "jax_compilation_cache_dir", os.environ.get("JAX_COMPILATION_CACHE_DIR", str(epath.Path("~/.cache/jax").expanduser()))
    )

    task_specs = _tasks.get_tasks(cfg.task_names)
    _tasks.check_layout(cfg.task_names, cfg.resolved_layout)
    norm_stats = pathlib.Path(cfg.assets_root) / cfg.asset_id / "norm_stats.json"
    if not norm_stats.exists():
        from lifedex import compute_norm_stats  # noqa: PLC0415

        logging.info(f"{norm_stats} not found; computing shared norm stats.")
        compute_norm_stats.main(
            compute_norm_stats.Args(
                stream=cfg.stream, tasks=cfg.tasks, layout=cfg.resolved_layout, regimes=cfg.regimes,
                assets_root=pathlib.Path(cfg.assets_root),
            )
        )
    logging.info(
        f"Tasks: {list(cfg.task_names)} | layout={cfg.resolved_layout} | regimes={cfg.regimes} | "
        f"norm stats={norm_stats} | init={cfg.base_params}"
    )
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = json.dumps(
        {**dataclasses.asdict(cfg), "resolved": {"layout": cfg.resolved_layout, "asset_id": cfg.asset_id,
                                                  "init_params": cfg.base_params, "tasks": list(cfg.task_names)}},
        indent=2,
        default=str,
    )
    config_file = run_dir / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != json.loads(cfg_json):
        logging.warning(f"Config differs from the one stored in {config_file}; the stored copy is kept as config.prev.json")
        (run_dir / "config.prev.json").write_text(config_file.read_text())
    config_file.write_text(cfg_json)

    writer = SummaryWriter(str(run_dir / "tb"))
    writer.add_text("config", f"```\n{cfg_json}\n```", 0)
    mesh = sharding.make_mesh(cfg.fsdp_devices)

    for k, task in enumerate(task_specs):
        if (done := _read_done(cfg, k)) is not None:
            logging.info(f"Stage {k} ({task.name}) already done: {done['checkpoint']}")
            continue
        run_stage(cfg, k, task_specs, mesh, writer)
        writer.flush()

    writer.close()
    logging.info(f"All {len(task_specs)} stages done. Run dir: {run_dir}")


if __name__ == "__main__":
    main(cli())
