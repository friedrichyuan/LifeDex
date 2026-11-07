#!/usr/bin/env bash
# Quick end-to-end check: a few steps per stage for every method on a 3-task stream (including fold_glasses and
# click_mouse, whose third-person camera key differs), with ER replay and EWC Fisher/penalty exercised.
#
#   bash lifedex/scripts/smoke_test.sh [seq_ft er ewc]
# Uses all visible GPUs; the batch size (BS, default 8) must be divisible by the number of GPUs.
# On an 8-GPU DLC node: STEPS=20 BS=16 bash lifedex/scripts/smoke_test.sh seq_ft er ewc
# Each invocation writes fresh runs to $LIFEDEX_CKPT_ROOT/smoke_<method>_<timestamp> (~25 GB per method).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHODS=("${@:-seq_ft er ewc}")
read -r -a METHODS <<< "${METHODS[*]}"
export LIFEDEX_CKPT_ROOT="${LIFEDEX_CKPT_ROOT:-$REPO/checkpoints/smoke}"
STAMP="$(date +%m%d_%H%M%S)"

for m in "${METHODS[@]}"; do
  bash "$REPO/lifedex/scripts/dlc_train.sh" "$m" "smoke_${m}_${STAMP}" \
    --tasks hammer_nail click_mouse fold_glasses \
    --steps-per-task "${STEPS:-6}" --batch-size "${BS:-8}" \
    --warmup-steps 2 --log-interval 2 --save-interval 1000 \
    --er.buffer-size 50 \
    --ewc.fisher-batches 3
done
