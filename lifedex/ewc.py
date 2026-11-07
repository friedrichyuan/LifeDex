"""Online Elastic Weight Consolidation (Kirkpatrick et al., 2017; Schwarz et al., 2018) for openpi nnx models.

Only trainable parameters (LoRA adapters, SigLIP, action projections) are regularized; frozen weights cannot drift.
The penalty gradient lam * F * (theta - theta*) is added analytically to the task gradient, which is identical to
differentiating lam / 2 * sum F (theta - theta*)^2 but avoids a second backward pass.
"""

import logging
import pathlib

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.utils as training_utils


def _leaves(tree):
    return jax.tree.leaves(tree)


def penalty_and_grad(params, fisher, anchor, lam: float):
    """Returns (sum F (p - a)^2, lam * F * (p - a)) with the structure of `params`."""
    diff = jax.tree.map(lambda p, a: p.astype(jnp.float32) - a, params, anchor)
    penalty = sum(jnp.sum(f * jnp.square(d)) for f, d in zip(_leaves(fisher), _leaves(diff), strict=True))
    grad = jax.tree.map(lambda f, d, p: (lam * f * d).astype(p.dtype), fisher, diff, params)
    return penalty, grad


def make_fisher_step(train_config):
    """Squared gradients of the training loss on one batch, for the trainable parameters."""

    def fisher_step(rng, state: training_utils.TrainState, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()
        observation, actions = batch

        def loss_fn(model, rng, observation, actions):
            return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

        diff_state = nnx.DiffState(0, train_config.trainable_filter)
        grads = nnx.grad(loss_fn, argnums=diff_state)(model, rng, observation, actions)
        return jax.tree.map(lambda g: jnp.square(g.astype(jnp.float32)), grads)

    return fisher_step


def estimate_fisher(fisher_step_jit, rng, state, loader, num_batches: int):
    fisher = None
    n = 0
    for i, (obs, actions, _) in enumerate(loader):
        sq = fisher_step_jit(jax.random.fold_in(rng, i), state, (obs, actions))
        fisher = sq if fisher is None else jax.tree.map(jnp.add, fisher, sq)
        n += 1
        if n >= num_batches:
            break
    if fisher is None:
        raise RuntimeError("Fisher estimation got no batches.")
    return jax.tree.map(lambda f: f / n, fisher)


def tree_mean(tree) -> jax.Array:
    leaves = _leaves(tree)
    return sum(jnp.sum(x) for x in leaves) / sum(x.size for x in leaves)


def normalize(fisher):
    mean = tree_mean(fisher)
    logging.info(f"Fisher mean before normalization: {float(mean):.3e}")
    return jax.tree.map(lambda f: f / (mean + 1e-30), fisher)


def _key(path: tuple) -> str:
    return "/".join(map(str, path))


def save(path: pathlib.Path, fisher) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = traverse_util.flatten_dict(fisher.to_pure_dict())
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{_key(k): np.asarray(jax.device_get(v)) for k, v in flat.items()})
    tmp.rename(path)


def load(path: pathlib.Path, template) -> nnx.State:
    """Loads a Fisher saved by `save` into a copy of `template` (an nnx.State with the trainable structure)."""
    template_keys = {_key(k): k for k in traverse_util.flatten_dict(template.to_pure_dict())}
    with np.load(path) as f:
        missing = set(template_keys) - set(f.files)
        if missing:
            raise ValueError(f"Fisher file {path} is missing {len(missing)} parameters, e.g. {sorted(missing)[:3]}")
        flat = {template_keys[k]: f[k] for k in template_keys}
    fisher = jax.tree.map(lambda x: x, template)  # copy structure
    fisher.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), fisher)
