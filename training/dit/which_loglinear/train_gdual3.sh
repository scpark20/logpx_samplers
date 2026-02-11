#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

STEPS=(3)
N_CLASSIFIERS=(3 7 11 15 19)

for s in "${STEPS[@]}"; do  
  for n in "${N_CLASSIFIERS[@]}"; do
    LOG="logs/dit/which_loglinear/s${s}_n${n}"

    # # 이미 로그 디렉토리가 있으면 스킵
    # if [[ -d "$LOG" ]]; then
    #   echo ">>> exists: $LOG — skipping"
    #   continue
    # fi

    echo ">>> n_steps=${s} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.dit.which_loglinear.train_gdual \
      --n_steps "$s" \
      --n_classifiers "$n" \
      --log_dir "$LOG"
  done
done