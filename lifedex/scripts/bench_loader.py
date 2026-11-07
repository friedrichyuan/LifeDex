"""Data-loader throughput benchmark (samples/s) for a few tasks and worker counts.

    PYTHONPATH=. python -m lifedex.scripts.bench_loader --workers 8 16 --tasks click_mouse fold_glasses
"""

import dataclasses
import time

import jax
import tyro

from lifedex import data as D
from lifedex import tasks as T
from lifedex.config import CLConfig


@dataclasses.dataclass
class Args:
    tasks: tuple[str, ...] = ("click_mouse", "fold_glasses")
    workers: tuple[int, ...] = (8, 16)
    batch_size: int = 32
    num_batches: int = 12


def main(args: Args):
    mesh = jax.sharding.Mesh(jax.devices(), ("B",))
    sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("B"))
    cfg = CLConfig(exp_name="bench", tasks=args.tasks)
    for name in args.tasks:
        ds, dc, _ = D.create_task_dataset(cfg, T.TASKS[name], init_params=cfg.base_params)
        for nw in args.workers:
            loader = D.build_loader(
                ds, dc, batch_size=args.batch_size, replay_ratio=0, seed=0, sharding=sh, num_workers=nw,
                num_batches=args.num_batches + 2,
            )
            it = iter(loader)
            next(it), next(it)  # warm-up: worker start-up
            t, n = time.time(), 0
            for _ in it:
                n += 1
            dt = time.time() - t
            print(f"RESULT task={name} workers={nw}: {n * args.batch_size / dt:.1f} samples/s", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
