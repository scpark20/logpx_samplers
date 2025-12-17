#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

STEPS=(9 3)
N_CLASSIFIERS=(18 16 14 12 10 8 6 4 2 0)

for s in "${STEPS[@]}"; do  
  for n in "${N_CLASSIFIERS[@]}"; do
    LOG="logs/gmdit/which_5k/s${s}_n${n}"

    # # 이미 로그 디렉토리가 있으면 스킵
    # if [[ -d "$LOG" ]]; then
    #   echo ">>> exists: $LOG — skipping"
    #   continue
    # fi

    echo ">>> n_steps=${s} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.gmdit.which_5k.train_gdual \
      --n_steps "$s" \
      --n_classifiers "$n" \
      --log_dir "$LOG"
  done
done