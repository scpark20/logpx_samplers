#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

STEPS=(9)
gamma_inits=(1.0)
tau_x_inits=(-2.0 2.0)
tau_e_inits=(-2.0 2.0)

for s in "${STEPS[@]}"; do
  for g in "${gamma_inits[@]}"; do
    for tx in "${tau_x_inits[@]}"; do
      for te in "${tau_e_inits[@]}"; do

        LOG="logs/gmdit/mobile_ll6_5k/s${s}_g${g}_tx${tx}_te${te}"

        # # 이미 로그 디렉토리가 있으면 스킵
        # if [[ -d "$LOG" ]]; then
        #   echo ">>> exists: $LOG — skipping"
        #   continue
        # fi

        echo ">>> n_steps=${s}, gamma=${g}, tau_x=${tx}, tau_e=${te} -> ${LOG}"
        mkdir -p "$LOG"

        python -m training.gmdit.mobile_ll6_5k.train_gdual \
          --n_steps "$s" \
          --gamma_init "$g" \
          --tau_x_init "$tx" \
          --tau_e_init "$te" \
          --log_dir "$LOG"

      done
    done
  done
done