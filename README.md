# LifeDex

Continual-learning baselines for the π0.5 VLA on [DexJoCo](https://github.com/brave-eai/dexjoco) dexterous
manipulation tasks: **sequential fine-tuning**, **experience replay** and **online EWC**. The setting follows
*Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning*
(Liu et al., 2026).

```
lifedex/   our code: training (train_cl.py), CL methods, evaluation (eval/), launch scripts (scripts/)
dexjoco/   submodule brave-eai/dexjoco@8d23b0f (simulator, openpi π0.5 fork), used unmodified
```

## Setup

```bash
git clone https://github.com/friedrichyuan/LifeDex.git && cd LifeDex
git submodule update --init dexjoco

(cd dexjoco/openpi && bash install.bash)   # base conda env `openpi`
bash lifedex/scripts/setup_env.sh          # env `lifedex`: clone of `openpi` + LeRobot 0.4.4 + TensorBoard
bash lifedex/scripts/setup_eval_env.sh     # env `dexjoco`: MuJoCo evaluation client
```

Scripts assume conda at `/mnt/data/yuanmingqi/miniconda3`; override with `export CONDA_ROOT=...`.
Data and weights go under `data/` (not tracked):

```bash
hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type=dataset --local-dir data/dexjoco_datasets
hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/*" --local-dir data/dexjoco_models
```

The PaliGemma tokenizer is downloaded to `.cache/openpi/` on first use; copy it there beforehand on offline nodes.

## Train

Single node, all visible GPUs. Rerunning a command resumes it; checkpoints and TensorBoard logs go to
`checkpoints/lifedex/<exp_name>/`.

```bash
bash lifedex/scripts/dlc_train.sh seq_ft seq_ft      --batch-size 64 --steps-per-task 15000 --warmup-steps 5000
bash lifedex/scripts/dlc_train.sh er     er_m1000    --batch-size 64 --steps-per-task 15000 --warmup-steps 5000 --er.buffer-size 1000
bash lifedex/scripts/dlc_train.sh ewc    ewc_lam1e-2 --batch-size 64 --steps-per-task 15000 --warmup-steps 5000 --ewc.lam 1e-2
```

## Evaluate

```bash
# 8 GPUs: policy servers on 0-5, MuJoCo rendering on 6-7
bash lifedex/scripts/dlc_eval.sh checkpoints/lifedex/seq_ft --gpus 0 1 2 3 4 5 --render-gpus 6 7
```

Evaluates every finished stage on the tasks learned so far (DexJoCo protocol: 3 seeds × 50 episodes) and writes
the success-rate matrix with ACC / NBT / normalized NBT to `checkpoints/lifedex/<exp_name>/eval/rand_obj/summary.md`.

## Setting

| | |
|---|---|
| Tasks | `hammer_nail → click_mouse → pick_bucket → water_plant → fold_glasses` (single-arm, `rand_obj`) |
| Model | π0.5 from `pi05_base`, LoRA, inputs/actions as in DexJoCo's single-task SFT |
| Per task | weights carried over, optimizer and LR schedule reset; one set of norm stats shared by all tasks |
| ER | fixed random buffer per past task (default 1000 samples), half of every batch is replay |
| EWC | online EWC, diagonal Fisher normalized to unit mean; λ needs a sweep |

Full details (options, measured throughput/latency, outputs, serving checkpoints): [lifedex/README.md](lifedex/README.md).
