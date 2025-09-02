#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

STEPS=(7 5)
N_CLASSIFIERS=(1 3 5 9)

for s in "${STEPS[@]}"; do
  for n in "${N_CLASSIFIERS[@]}"; do
    LOG="logs/gmdit/multi/s${s}_n${n}"

    # # 이미 로그 디렉토리가 있으면 스킵
    # if [[ -d "$LOG" ]]; then
    #   echo ">>> exists: $LOG — skipping"
    #   continue
    # fi

    echo ">>> n_steps=${s} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.gmdit.multi.train_gdual \
      --n_steps "$s" \
      --log_dir "$LOG"
  done
done
