#!/usr/bin/env bash
set -e
export DPM_TQDM=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# 하나만 고르세요: none | except_gamma | except_tau | except_kappa | shared_taukappa
FLAG=${FLAG:-except_tau}

case "$FLAG" in
  none)             SUF="base"; FLAGS="";;
  except_gamma)     SUF="eg";   FLAGS="--except_gamma";;
  except_tau)       SUF="et";   FLAGS="--except_tau";;
  except_kappa)     SUF="ek";   FLAGS="--except_kappa";;
  shared_taukappa)  SUF="st";   FLAGS="--shared_taukappa";;
  *) echo "unknown FLAG=$FLAG (use: none|except_gamma|except_tau|except_kappa|shared_taukappa)"; exit 1;;
esac

for s in 3 5 7 9; do
  LOG="logs/ablations/params/s${s}_${SUF}"

  if [[ -d "$LOG" ]]; then
    echo ">>> exists: $LOG — skipping"
    continue
  fi

  echo ">>> n_steps=${s}, FLAG=${FLAG} (${FLAGS:-no-flag}) -> ${LOG}"

  python -u -m training.ablations.params.train_gdual \
    --n_steps "${s}" \
    --log_dir "${LOG}" \
    ${FLAGS}
done
