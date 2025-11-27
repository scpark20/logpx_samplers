#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

STEPS=(4 3 2)

for s in "${STEPS[@]}"; do
  LOG="logs/gmdit/epd_traj_edm_5e3_10k_re2/s${s}"

  # # 이미 로그 디렉토리가 있으면 스킵
  # if [[ -d "$LOG" ]]; then
  #   echo ">>> exists: $LOG — skipping"
  #   continue
  # fi

  echo ">>> n_steps=${s} -> ${LOG}"
  mkdir -p "$LOG"

  python -m training.gmdit.epd_traj_edm_5e3_10k_re2.train_epd \
    --n_steps "$s" \
    --log_dir "$LOG"
done