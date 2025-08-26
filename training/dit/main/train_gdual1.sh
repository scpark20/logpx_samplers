#!/usr/bin/env bash
set -e
export DPM_TQDM=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

STEPS=(5 7)

for s in "${STEPS[@]}"; do
  LOG="logs/dit/s${s}"

  # 이미 로그 디렉토리가 있으면 스킵
  if [[ -d "$LOG" ]]; then
    echo ">>> exists: $LOG — skipping"
    continue
  fi

  echo ">>> n_steps=${s} -> ${LOG}"

  python -m training.dit.main.train_gdual \
    --n_steps "$s" \
    --log_dir "$LOG"
done