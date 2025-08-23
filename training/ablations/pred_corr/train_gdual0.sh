#!/usr/bin/env bash
set -e
export DPM_TQDM=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for s in 2 8; do
  for combo in p1 p1c1 p1c2 p2c2 p2c3 p3c3; do
    case "$combo" in
      p1)   P=1; C=1; UC=0 ;;
      p1c1) P=1; C=1; UC=1 ;;
      p1c2) P=1; C=2; UC=1 ;;
      p2c2) P=2; C=2; UC=1 ;;
      p2c3) P=2; C=3; UC=1 ;;
      p3c3) P=3; C=3; UC=1 ;;
      *) echo "unknown combo: $combo"; exit 1 ;;
    esac

    LOG="logs/ablations/pred_corr/${combo}_s${s}"
    echo ">>> n_steps=${s}, pred=${P}, corr=${C}, corrector=$UC -> ${LOG}"

    if [[ $UC -eq 1 ]]; then
      python -m training.ablations.pred_corr.train_gdual \
        --n_steps "$s" --pred_order "$P" --corr_order "$C" \
        --log_dir "$LOG" --use_corrector
    else
      python -m training.ablations.pred_corr.train_gdual \
        --n_steps "$s" --pred_order "$P" --corr_order "$C" \
        --log_dir "$LOG"
    fi
  done
done
