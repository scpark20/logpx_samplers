#!/usr/bin/env bash
set -euo pipefail

# 기본값 설정 (이미 설정되어 있으면 그대로 사용)
export DPM_TQDM="${DPM_TQDM:-False}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

TOTAL_STEPS=(2 3 6 7 10)
STEPS=(9 3)

for t in "${TOTAL_STEPS[@]}"; do
  for s in "${STEPS[@]}"; do
    LOG="logs/gmdit/mobile_total/s${s}_t${t}"

    # 이미 로그 디렉토리가 있으면 스킵하려면 아래 주석 해제
    # if [[ -d "$LOG" ]]; then
    #   echo ">>> exists: $LOG — skipping"
    #   continue
    # fi

    echo ">>> total_steps=${t}, n_steps=${s} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.gmdit.mobile_total.train_gdual \
      --n_steps "${s}" \
      --log_dir "${LOG}" \
      --total_steps "${t}"
  done
done