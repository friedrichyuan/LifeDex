"""Data pipeline: unified single-arm/bimanual layout, experience-replay buffer and mixed-batch loader."""

from collections.abc import Iterator, Sequence
import dataclasses
import json
import logging
import multiprocessing
import pathlib
import typing

import jax
import numpy as np
import torch
from typing_extensions import override

from lifedex import tasks as _tasks
from lifedex.config import _config

import openpi.models.model as _model  # noqa: E402
from openpi.policies import dual_arm_policy  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402


# --------------------------------------------------------------------------------------------------------------------
# Transforms / data config
# --------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PadStateAction(_transforms.DataTransformFn):
    """Zero-pads state/actions to the unified bimanual dims *before* normalization (train and inference time)."""

    state_dim: int = _tasks.LAYOUTS["bimanual"][0]
    action_dim: int = _tasks.LAYOUTS["bimanual"][1]

    def __call__(self, data: dict) -> dict:
        data["state"] = _transforms.pad_to_dim(np.asarray(data["state"]), self.state_dim, axis=-1)
        if "actions" in data:
            data["actions"] = _transforms.pad_to_dim(np.asarray(data["actions"]), self.action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class UnifiedDexDataConfig(_config.DataConfigFactory):
    """DexJoCo LeRobot dataset of one task, mapped into the unified bimanual layout (see lifedex/tasks.py).
    Single-arm streams use dexjoco's own `SingleArmDataConfig` instead."""

    root: pathlib.Path | None = None
    task: _tasks.TaskSpec | None = None
    # Third-person camera key (differs between rand_obj and rand_full); None -> task.base_img.
    base_img: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        assert self.task is not None
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "base": self.base_img or self.task.base_img,
                    "wrist_left": self.task.wrist_left_img,
                    "wrist_right": self.task.wrist_right_img,
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[dual_arm_policy.DualArmInputs(model_type=model_config.model_type), PadStateAction()],
            outputs=[dual_arm_policy.DualArmOutputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
            root=self.root,
        )


def create_task_dataset(cfg, task: _tasks.TaskSpec, *, init_params: str):
    """Fully transformed (normalized, tokenized) dataset of `task`: the concatenation of its datasets in every training
    regime of `cfg` (e.g. rand_obj + rand_full). Returns (dataset, data_config, train_config of the first regime)."""
    from lifedex.config import make_train_config  # noqa: PLC0415

    parts, first = [], None
    for regime in cfg.regimes:
        train_config = make_train_config(cfg, task, init_params=init_params, regime=regime)
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        ds = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
        parts.append(_data_loader.transform_dataset(ds, data_config))
        if first is None:
            first = (data_config, train_config)
    dataset = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
    return dataset, first[0], first[1]


# --------------------------------------------------------------------------------------------------------------------
# Experience replay
# --------------------------------------------------------------------------------------------------------------------
def replay_indices(num_frames: int, buffer_size: float, seed: int) -> np.ndarray:
    """Frame indices kept in the replay buffer for one task (fixed for the whole run, as in ER/LIBERO)."""
    n = round(buffer_size * num_frames) if 0 < buffer_size < 1 else int(buffer_size)
    n = max(1, min(n, num_frames))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_frames, size=n, replace=False))


def save_replay_manifest(path: pathlib.Path, manifest: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.tolist() for k, v in manifest.items()}))


class MixedDataset(torch.utils.data.Dataset):
    """Concatenation of [current task, *replay tasks]; each sample is tagged with `is_replay`."""

    def __init__(self, current: _data_loader.Dataset, replay: Sequence[_data_loader.Dataset] = ()):
        self._parts = [current, *replay]
        self._offsets = np.cumsum([0] + [len(p) for p in self._parts])

    def offset(self, part: int) -> int:
        return int(self._offsets[part])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index):
        index = int(index)
        part = int(np.searchsorted(self._offsets, index, side="right") - 1)
        sample = self._parts[part][index - self._offsets[part]]
        sample["is_replay"] = np.bool_(part > 0)
        return sample


class MixedBatchSampler(torch.utils.data.Sampler):
    """Yields batches with `batch_size - num_replay` current-task samples (epoch-wise shuffled, no replacement)
    and `num_replay` samples drawn uniformly with replacement from the replay pool."""

    def __init__(
        self,
        num_current: int,
        replay_pool: np.ndarray,
        batch_size: int,
        num_replay: int,
        seed: int,
        num_batches: int | None = None,
    ):
        if num_replay > 0 and len(replay_pool) == 0:
            raise ValueError("num_replay > 0 but the replay pool is empty.")
        self._num_current = num_current
        self._replay_pool = replay_pool
        self._batch_size = batch_size
        self._num_replay = num_replay
        self._num_cur = batch_size - num_replay
        self._seed = seed
        self._num_batches = num_batches
        if self._num_cur > num_current:
            raise ValueError(f"Current-task part of the batch ({self._num_cur}) exceeds dataset size ({num_current}).")

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self._seed)
        produced = 0
        while True:
            perm = rng.permutation(self._num_current)
            for start in range(0, len(perm) - self._num_cur + 1, self._num_cur):
                if self._num_batches is not None and produced >= self._num_batches:
                    return
                idx = perm[start : start + self._num_cur].tolist()
                if self._num_replay:
                    idx += rng.choice(self._replay_pool, size=self._num_replay, replace=True).tolist()
                produced += 1
                yield idx

    def __len__(self) -> int:
        return self._num_batches if self._num_batches is not None else 2**62


class CLDataLoader:
    """Iterates (Observation, actions, is_replay) as globally-sharded jax arrays."""

    def __init__(
        self,
        dataset: MixedDataset,
        sampler: MixedBatchSampler,
        data_config: _config.DataConfig,
        *,
        sharding: jax.sharding.Sharding,
        num_workers: int,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Multi-process (multi-node) data loading is not supported.")
        self._data_config = data_config
        self._sharding = sharding
        self._loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
            persistent_workers=False,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
            worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
        )

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._loader:
            is_replay = batch.pop("is_replay")
            batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
            is_replay = jax.make_array_from_process_local_data(self._sharding, is_replay)
            yield _model.Observation.from_dict(batch), batch["actions"], is_replay


def build_loader(
    current: _data_loader.Dataset,
    data_config: _config.DataConfig,
    *,
    replay: Sequence[tuple[_data_loader.Dataset, np.ndarray]] = (),
    batch_size: int,
    replay_ratio: float,
    seed: int,
    sharding: jax.sharding.Sharding,
    num_workers: int,
    num_batches: int | None = None,
) -> CLDataLoader:
    """`replay` is a list of (task dataset, kept frame indices) for the previous tasks."""
    dataset = MixedDataset(current, [ds for ds, _ in replay])
    pool = np.concatenate([idx + dataset.offset(i + 1) for i, (_, idx) in enumerate(replay)]) if replay else np.array([])
    num_replay = round(batch_size * replay_ratio) if replay else 0
    if replay:
        logging.info(
            f"Replay pool: {len(pool)} samples from {len(replay)} task(s); {num_replay}/{batch_size} per batch."
        )
    sampler = MixedBatchSampler(len(current), pool.astype(np.int64), batch_size, num_replay, seed, num_batches)
    return CLDataLoader(dataset, sampler, data_config, sharding=sharding, num_workers=num_workers)
