#!/usr/bin/env bash
set -euo pipefail
export DPM_TQDM=${DPM_TQDM:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

STEPS=(3 5 7 9)
KS=(0.5)

for s in "${STEPS[@]}"; do
  for k in "${KS[@]}"; do
    LOG="logs/dit/bns/vec/s${s}_k${k}"

    # 이미 로그 디렉토리가 있으면 스킵
    if [[ -d "$LOG" ]]; then
      echo ">>> exists: $LOG — skipping"
      continue
    fi

    echo ">>> n_steps=${s}, k=${k} -> ${LOG}"
    mkdir -p "$LOG"

    python -m training.dit.bns.vec.train_bns \
      --n_steps "$s" \
      --k "$k" \
      --log_dir "$LOG"
  done
done