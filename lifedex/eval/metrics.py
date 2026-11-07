"""Continual-learning metrics on the success-rate matrix.

R[k][j] = success rate on task j evaluated with the checkpoint after stage k (rows: checkpoints, cols: tasks), as in
Fig. 1 of Liu et al. (2026). Only entries that were evaluated are used, so partially trained runs can be summarized.

* NBT (negative backward transfer, LIBERO / Liu et al.): for task j, the mean over later stages tau > j of
  R[j][j] - R[tau][j]; averaged over tasks that have at least one later stage. > 0 means forgetting.
* NBT_norm (Liu et al., App. A.2): the same with each term divided by R[j][j] (tasks with R[j][j] = 0 are skipped),
  so complete forgetting is 1 regardless of the initial success rate.
* ACC: mean success over all learned tasks after the last evaluated stage, mean_j R[K][j], j <= K.
* Plasticity: mean of the diagonal, mean_k R[k][k] (how well each task is learned when it is trained).
Each metric is computed per evaluation seed and reported as mean +- std over seeds.
"""

import warnings

import numpy as np

# pi0.5 single-task SFT success rates (%) from DexJoCo Table II (mean over 3 seeds x 50 episodes).
DEXJOCO_PI05_SINGLE_TASK = {
    "rand_obj": {
        "hammer_nail": 84.7, "click_mouse": 64.7, "pick_bucket": 84.0, "pinch_tongs": 24.0,
        "fold_glasses": 72.0, "water_plant": 88.7,
    },
    "rand_full": {
        "hammer_nail": 17.3, "click_mouse": 54.7, "pick_bucket": 78.7, "pinch_tongs": 18.7,
        "fold_glasses": 39.3, "water_plant": 75.3,
    },
}


def metrics_from_matrix(R: np.ndarray) -> dict[str, float]:
    """R: (K, K) with NaN for missing entries."""
    K = R.shape[0]
    rows = [k for k in range(K) if not np.all(np.isnan(R[k, : k + 1]))]
    out: dict[str, float] = {}
    if not rows:
        return out
    last = max(rows)
    out["ACC"] = float(np.nanmean(R[last, : last + 1]))
    diag = [R[k, k] for k in range(K) if not np.isnan(R[k, k])]
    if diag:
        out["plasticity"] = float(np.mean(diag))
    nbt, nbt_norm = [], []
    for j in range(K):
        if np.isnan(R[j, j]):
            continue
        later = [R[t, j] for t in range(j + 1, K) if not np.isnan(R[t, j])]
        if not later:
            continue
        drops = [R[j, j] - x for x in later]
        nbt.append(np.mean(drops))
        if R[j, j] > 0:
            nbt_norm.append(np.mean(drops) / R[j, j])
    if nbt:
        out["NBT"] = float(np.mean(nbt))
    if nbt_norm:
        out["NBT_norm"] = float(np.mean(nbt_norm))
    return out


def summarize(results: dict, task_names: list[str], seeds: list[int]) -> dict:
    """results[(stage, task, seed)] = success rate in [0, 1]. Returns matrix mean/std and metrics mean/std."""
    K = len(task_names)
    idx = {t: j for j, t in enumerate(task_names)}
    per_seed = np.full((len(seeds), K, K), np.nan)
    for (stage, task, seed), sr in results.items():
        if seed in seeds and task in idx:
            per_seed[seeds.index(seed), stage, idx[task]] = sr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices for unevaluated entries
        mean = np.nanmean(per_seed, axis=0)
        std = np.nanstd(per_seed, axis=0)
    n = np.sum(~np.isnan(per_seed), axis=0)
    # Metrics per seed, only over entries every seed has (consistent support across seeds).
    complete = n == len(seeds)
    seed_metrics = []
    for s in range(len(seeds)):
        R = np.where(complete, per_seed[s], np.nan)
        seed_metrics.append(metrics_from_matrix(R))
    keys = sorted({k for m in seed_metrics for k in m})
    metrics = {
        k: (float(np.mean([m[k] for m in seed_metrics if k in m])), float(np.std([m[k] for m in seed_metrics if k in m])))
        for k in keys
    }
    return {"mean": mean, "std": std, "n_seeds": n, "metrics": metrics}


def to_markdown(summary: dict, task_names: list[str], *, regime: str, title: str) -> str:
    K = len(task_names)
    mean, std, n = summary["mean"], summary["std"], summary["n_seeds"]
    lines = [f"# {title}", "", f"Success rate (%), mean ± std over evaluation seeds; regime `{regime}`.", ""]
    lines.append("| checkpoint \\ task | " + " | ".join(task_names) + " |")
    lines.append("|---|" + "---|" * K)
    for k in range(K):
        if np.all(n[k] == 0):
            continue
        cells = []
        for j in range(K):
            if n[k, j] == 0:
                cells.append("")
            else:
                cells.append(f"{100 * mean[k, j]:.1f} ± {100 * std[k, j]:.1f}" + ("" if n[k, j] > 1 else "*"))
        lines.append(f"| after stage {k} ({task_names[k]}) | " + " | ".join(cells) + " |")
    ref = DEXJOCO_PI05_SINGLE_TASK.get(regime, {})
    lines.append("| DexJoCo π0.5 single-task (Table II) | " + " | ".join(str(ref.get(t, "")) for t in task_names) + " |")
    lines += ["", "`*` = single seed.", "", "| metric | value |", "|---|---|"]
    for key, (m, s) in summary["metrics"].items():
        scale = 100 if key in ("ACC", "plasticity", "NBT") else 1
        unit = " (%)" if scale == 100 else ""
        lines.append(f"| {key}{unit} | {scale * m:.2f} ± {scale * s:.2f} |")
    lines += [
        "",
        "ACC: mean over learned tasks after the last evaluated stage. Plasticity: mean of the diagonal. "
        "NBT: mean drop from R[j][j] over later stages (> 0 = forgetting). NBT_norm: NBT relative to R[j][j].",
    ]
    return "\n".join(lines) + "\n"
