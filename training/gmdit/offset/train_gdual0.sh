#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

STEPS=(9)
OFFSETS=(2 3)

for s in "${STEPS[@]}"; do
  for offset in "${OFFSETS[@]}"; do
    LOG="logs/gmdit/offset/s${s}_offset${offset}"

    # # 이미 로그 디렉토리가 있으면 스킵
    # if [[ -d "$LOG" ]]; then
    #   echo ">>> exists: $LOG — skipping"
    #   continue
    # fi

    echo ">>> n_steps=${s} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.gmdit.offset.train_gdual \
      --n_steps "$s" \
      --offset "$offset" \
      --log_dir "$LOG"
  done
done