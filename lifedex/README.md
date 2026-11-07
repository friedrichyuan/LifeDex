# LifeDex — continual learning for dexterous VLAs

Continual-learning baselines for π0.5 on [DexJoCo](../dexjoco), following the setting of
*Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning*
(Liu et al., 2026). The code lives here and only *imports* `dexjoco/openpi`; nothing under `dexjoco/` is modified.

## Setting

| | |
|---|---|
| Task stream `dex5` | `hammer_nail → click_mouse → pick_bucket → water_plant → fold_glasses` (all single-arm) |
| Model | π0.5 `pi05_base`, PaliGemma LoRA + action-expert LoRA — identical to dexjoco single-task SFT |
| Layout `single_arm` | 22-dim action `[xyz, rotvec, hand16]`, 23-dim state `[xyz, quat, hand16]` (absolute, EEF + hand joints), padded to the model's 32 dims; cameras: third-person → base, wrist → left wrist, right wrist masked (dexjoco `SingleArmDataConfig`) |
| Data | `rand_obj` only (default). DexJoCo trains and evaluates each regime separately (`--regimes rand_full` for the other one; several regimes are concatenated per task) |
| Norm stats | one shared set over all tasks of the stream (`lifedex/assets/<stream>_<layout>_<regimes>`), computed automatically on first run |
| Per task | weights carried over; fresh AdamW + LR schedule; `steps_per_task` steps (default 30k, bs 32, lr 5e-5) |

Streams mixing in bimanual tasks (e.g. `--stream dex5_bi`) automatically use the `bimanual` layout
(`pi05_base_action_dim_44`, state/action zero-padded to 46/44, dexjoco `multi_task` convention).

π0.5 single-task SFT success rates reported by DexJoCo (Table II, rand_obj / rand_full, %):
hammer_nail 84.7 / 17.3, click_mouse 64.7 / 54.7, pick_bucket 84.0 / 78.7, water_plant 88.7 / 75.3,
fold_glasses 72.0 / 39.3.

Methods:

* **`seq_ft`** — sequential fine-tuning on the current task only.
* **`er`** — experience replay. After each task a fixed random subset of `--er.buffer-size` samples
  (frame + action chunk; default 1000 ≈ 2–5% of a DexJoCo task) is kept; every batch is
  `1 - replay_ratio` current task and `replay_ratio` (default 0.5, i.e. 1:1 as in LIBERO) uniformly from all buffers.
* **`ewc`** — online EWC on the trainable parameters. After each task: empirical diagonal Fisher on
  `--ewc.fisher-batches` batches, normalized to unit mean, accumulated `F ← γF + F_k`; anchor = end-of-task weights.
  The penalty `λ/2 Σ F (θ−θ*)²` is added analytically to the gradient (before global-norm clipping).
  **λ needs a sweep**: with the normalized Fisher, λ=1 dominates the task gradient 35–150× (see `ewc_grad_norm`
  vs `task_grad_norm` in TensorBoard). Default λ=1e-2.

## Setup (once)

```bash
git submodule update --init dexjoco          # dexjoco is a submodule (pinned to brave-eai/dexjoco@8d23b0f)
bash lifedex/scripts/setup_env.sh            # clones conda env `openpi` -> `lifedex` and installs deps
bash lifedex/scripts/setup_eval_env.sh       # conda env `dexjoco` for the MuJoCo evaluation client
```

The original `openpi` env is not touched (it belongs to another project). The PaliGemma tokenizer is expected in
`.cache/openpi/big_vision/paligemma_tokenizer.model` (copied from the shared NAS cache) so that jobs run offline.

## Training

```bash
# DLC: single node, all visible GPUs (data parallel). Rerunning the same command resumes.
bash lifedex/scripts/dlc_train.sh seq_ft seq_ft
bash lifedex/scripts/dlc_train.sh er     er_m1000    --er.buffer-size 1000
bash lifedex/scripts/dlc_train.sh ewc    ewc_lam1e-2 --ewc.lam 1e-2
```

Any field of `lifedex/config.py:CLConfig` can be overridden (`--steps-per-task`, `--batch-size`, `--peak-lr`,
`--tasks a b c`, `--regimes rand_full`, `--er.replay-ratio`, `--ewc.gamma`, ...). Env vars: `LIFEDEX_CKPT_ROOT`
(default `checkpoints/lifedex`), `LIFEDEX_DATA_ROOT` (default `data`), `NUM_WORKERS`, `CONDA_ROOT`, `ENV_NAME`.

**Data loading is the bottleneck on fast GPUs** (AV1 video decoding, ~15–20 samples/s per worker): the 4×H800
smoke test ran at 3.6 steps/s × bs16 ≈ 58 samples/s with 4 workers. `dlc_train.sh` therefore defaults to
`nproc - 4` workers (max 64); request plenty of CPU cores for the DLC job. Loader throughput:
`PYTHONPATH=. python -m lifedex.scripts.bench_loader --workers 8 16 32`.

Quick check (a few steps per stage, 3 tasks, all visible GPUs): `bash lifedex/scripts/smoke_test.sh seq_ft er ewc`
(`STEPS=20 BS=16` on a multi-GPU node; writes fresh runs to `checkpoints/smoke/smoke_<method>_<timestamp>`).

Monitoring: `tensorboard --logdir checkpoints/lifedex` — `train/loss`, `train/loss_current`, `train/loss_replay`,
`train/ewc_penalty`, `train/ewc_grad_norm`, `train/task_grad_norm`, `train/lr`, `train/steps_per_sec`,
`train/stage`, camera views per stage. The x-axis is the global step (`stage * steps_per_task + step`).

## Evaluation

Uses dexjoco's stock client (`dexjoco-openpi-eval`, conda env `dexjoco`, created by
`lifedex/scripts/setup_eval_env.sh`) against LifeDex policy servers (`lifedex.eval.serve`, env `lifedex`).

```bash
# All finished stages, DexJoCo protocol (3 seeds x 50 episodes per (checkpoint, learned task)).
# 8-GPU node: policy servers on 6 GPUs, MuJoCo rendering on the other 2 (see latency notes below).
bash lifedex/scripts/dlc_eval.sh checkpoints/lifedex/seq_ft --gpus 0 1 2 3 4 5 --render-gpus 6 7
# quick check
bash lifedex/scripts/dlc_eval.sh checkpoints/lifedex/seq_ft --seeds 0 --episodes 5 --out-dir /tmp/quick_eval
# re-summarize existing results
bash lifedex/scripts/dlc_eval.sh checkpoints/lifedex/seq_ft --summarize-only
```

* Work units are (stage, task, seed); each GPU slot keeps one policy server per stage and reuses it. Finished
  units are skipped on rerun, so evaluation can be run while training continues (it picks up newly finished stages).
* `--scope seen` (default) evaluates checkpoint k on tasks 0..k (lower triangle = what NBT needs); `--scope all`
  also evaluates future tasks. `--regime rand_full` evaluates the other regime.
* Output: `<run_dir>/eval/<regime>/summary.md` (matrix mean ± std over seeds, DexJoCo single-task reference row,
  ACC / plasticity / NBT / NBT_norm with std over seeds), `results.json`, per-unit `client.log` and videos (seed 0).
* Protocol follows DexJoCo §IV: asynchronous chunked inference (next chunk is computed while the current one
  executes, overlapping chunks blended), π0.5 chunk 30, 30 Hz control, simulation paced to ≤ real time. DexJoCo
  deploys on an RTX 4090 with π0.5 latency ≈ 85 ms.
* Measured on an H20 box: `env.step` ≈ 37 ms (4 × 640² EGL renders ≈ 33 ms, physics ≈ 4 ms); with the client's
  resizing + video encoding the loop runs at ~15 Hz (a 1200-step episode takes ~80 s). CPU software rendering
  (Mesa llvmpipe) is 15–20× slower (≈ 0.5–0.7 s per step) and therefore not an option.
* Inference latency (`INFER_STATS` in `servers/*.log`): ≈ 95 ms on an otherwise idle GPU; 150 ms with one
  EGL-rendering client on the same GPU, 250 ms with three (graphics/compute time-slicing); two servers on one GPU
  ≈ 180 ms each. Hence one server per GPU and rendering on separate GPUs (`--render-gpus`). Keep the evaluation
  hardware and layout identical across methods.
* Cost: the full lower triangle of the 5-task stream is 15 entries × 3 seeds × 50 episodes = 2,250 episodes
  (~30 slot-hours); with 6 server slots ≈ 5 h per run. Budget ~2–3 CPU cores per slot.

## Outputs

```
<ckpt_root>/<exp_name>/
  config.json, tb/
  stage00_hammer_nail/checkpoints/<last_step>/{params,train_state,assets}   # servable
  stage00_hammer_nail/DONE.json                                             # stage complete
  stage01_click_mouse/replay_manifest.json   (er)  frame indices kept per previous task
  stage00_hammer_nail/ewc_fisher.npz         (ewc) accumulated Fisher
```

Each stage checkpoint can be served for any task of the stream (task identity comes from the prompt):

```python
from lifedex.config import CLConfig, make_train_config
from lifedex import tasks
from openpi.policies import policy_config
tc = make_train_config(CLConfig(exp_name="x"), tasks.TASKS["hammer_nail"], init_params="unused")
policy = policy_config.create_trained_policy(tc, "<stage dir>/checkpoints/<step>")
# obs (single_arm layout): base, wrist (uint8 224x224), state (23), prompt  -> actions (30, 22)
```

Evaluation: see the Evaluation section (single-arm layout uses `dexjoco/configs/<regime>/<task>.yaml`, the same
client and configs as dexjoco's single-task checkpoints).
