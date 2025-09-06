#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

STEPS=(3 4)
N_CLIPS=(0 1 2 3 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19)

for s in "${STEPS[@]}"; do
  for n in "${N_CLIPS[@]}"; do
    LOG="logs/sana/which/s${s}_n${n}"

    # 이미 로그 디렉토리가 있으면 스킵
    if [[ -d "$LOG" ]]; then
      echo ">>> exists: $LOG — skipping"
      continue
    fi

    echo ">>> n_steps=${s}, n_clips=${n} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.sana.which.train_gdual \
      --n_steps "$s" \
      --n_clips "$n" \
      --log_dir "$LOG"
  done
done