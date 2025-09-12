#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

#STEPS=(3 5 7 9)
STEPS=(8 6 4)

for s in "${STEPS[@]}"; do
  LOG="logs/dit/bns_mobile/s${s}"

  # # 이미 로그 디렉토리가 있으면 스킵
  # if [[ -d "$LOG" ]]; then
  #   echo ">>> exists: $LOG — skipping"
  #   continue
  # fi

  echo ">>> n_steps=${s} -> ${LOG}"
  mkdir -p "$LOG"

  python -m training.dit.bns_mobile.train_bns \
    --n_steps "$s" \
    --log_dir "$LOG"
done