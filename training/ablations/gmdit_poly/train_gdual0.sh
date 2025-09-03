#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
POLY_ORDER=4
LOG="logs/ablations/gmdit_poly${POLY_ORDER}"

# # 이미 로그 디렉토리가 있으면 스킵
# if [[ -d "$LOG" ]]; then
#   echo ">>> exists: $LOG — skipping"
#   continue
# fi

echo ">>> ${LOG}"
mkdir -p "$LOG"

python -m training.ablations.gmdit_poly.train_gdual \
  --log_dir "$LOG" \
  --poly_order "$POLY_ORDER"