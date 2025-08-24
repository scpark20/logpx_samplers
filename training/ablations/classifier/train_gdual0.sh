#!/usr/bin/env bash
set -e
export DPM_TQDM=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

S=5

for i in $(seq 0 19); do
  LOG="logs/ablations/classifier/s${S}/clf${i}"

  if [[ -d "$LOG" ]]; then
    echo ">>> exists: $LOG — skipping"
    continue
  fi

  echo ">>> n_steps=${S}, classifier_num=${i} -> ${LOG}"

  python -u -m training.ablations.classifier.train_gdual \
    --n_steps "${S}" \
    --classifier_num "${i}" \
    --log_dir "${LOG}"
done
