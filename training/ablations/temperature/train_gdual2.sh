#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

STEPS=(7 5)
TEMPS=(2.0)

for s in "${STEPS[@]}"; do
  for t in "${TEMPS[@]}"; do
    LOG="logs/ablations/temperature/s${s}_t${t}"

    # 이미 로그 디렉토리가 있으면 스킵
    if [[ -d "$LOG" ]]; then
      echo ">>> exists: $LOG — skipping"
      continue
    fi

    echo ">>> n_steps=${s}, T=${t} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.ablations.temperature.train_gdual \
      --n_steps "$s" \
      --log_dir "$LOG" \
      --temperature "$t"
  done
done
