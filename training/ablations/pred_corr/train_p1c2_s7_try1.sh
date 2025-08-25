#!/usr/bin/env bash
set -e
export DPM_TQDM=False
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

for s in 7; do
  for combo in p1c2; do
    case "$combo" in
      p1)   P=1; C=1; UC=0 ;;
      p1c2) P=1; C=2; UC=1 ;;
      p2)   P=2; C=2; UC=0 ;;
      p2c2) P=2; C=2; UC=1 ;;
      p2c3) P=2; C=3; UC=1 ;;
      *) echo "unknown combo: $combo"; exit 1 ;;
    esac

    LOG="logs/ablations/pred_corr/${combo}_s${s}_try1"

    # 이미 로그 디렉토리가 있으면 스킵
    if [[ -d "$LOG" ]]; then
      echo ">>> exists: $LOG — skipping"
      continue
    fi

    echo ">>> n_steps=${s}, pred=${P}, corr=${C}, corrector=${UC} -> ${LOG}"

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
