#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

STEPS=(9 7 5 3)

for s in "${STEPS[@]}"; do
  LOG="logs/dit/traj_loglinear/s${s}"

  # # 이미 로그 디렉토리가 있으면 스킵
  # if [[ -d "$LOG" ]]; then
  #   echo ">>> exists: $LOG — skipping"
  #   continue
  # fi

  echo ">>> n_steps=${s} -> ${LOG}"
  mkdir -p "$LOG"

  python -m training.dit.traj_loglinear.train_gdual \
    --n_steps "$s" \
    --log_dir "$LOG"
done