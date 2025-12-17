#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

#STEPS=(3 4 5 6 7 8)
STEPS=(4 6 7)
gamma_inits=(0.0)
tau_inits=(1.0)

for s in "${STEPS[@]}"; do
  for g in "${gamma_inits[@]}"; do
    for t in "${tau_inits[@]}"; do

      LOG="logs/gmdit/mobile_ll5_5k/s${s}_g${g}_t${t}"

      # # 이미 로그 디렉토리가 있으면 스킵
      # if [[ -d "$LOG" ]]; then
      #   echo ">>> exists: $LOG — skipping"
      #   continue
      # fi

      echo ">>> n_steps=${s}, gamma=${g}, tau=${t} -> ${LOG}"
      mkdir -p "$LOG"

      python -m training.gmdit.mobile_ll5_5k.train_gdual \
        --n_steps "$s" \
        --gamma_init "$g" \
        --tau_init "$t" \
        --log_dir "$LOG"

    done
  done
done