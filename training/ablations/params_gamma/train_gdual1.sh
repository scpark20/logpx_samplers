#!/usr/bin/env bash
set -e
export DPM_TQDM=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

FLAG=${FLAG:-gamma_n1}

case "$FLAG" in
  none)             SUF="base"; FLAGS="";;
  gamma_0)     SUF="g0";   FLAGS="--gamma_0";;
  gamma_n1)       SUF="gn";   FLAGS="--gamma_n1";;
  *) echo "unknown FLAG=$FLAG"; exit 1;;
esac

for s in 9 7 5 3; do
  LOG="logs/ablations/params_gamma/s${s}_${SUF}"

  # if [[ -d "$LOG" ]]; then
  #   echo ">>> exists: $LOG — skipping"
  #   continue
  # fi

  echo ">>> n_steps=${s}, FLAG=${FLAG} (${FLAGS:-no-flag}) -> ${LOG}"

  python -u -m training.ablations.params_gamma.train_gdual \
    --n_steps "${s}" \
    --log_dir "${LOG}" \
    ${FLAGS}
done
