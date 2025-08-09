#!/usr/bin/env bash
set -e

TAG=dit_train
MODEL=DiT
DATA=ImageNet
SAVE_ROOT=samplings/dit/train
BATCH_SIZE=5
ALGO=data_prediction
SKIP=time_uniform
ORDER=1
N_SAMPLES=10000
SEED_OFFSET=1

SOLVERS=("Euler")
NFES=(200)
CFGS=(1.375)

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      echo "▶ Running ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg}"
      python -m runs.sample \
        --tag "$TAG" \
        --model "$MODEL" \
        --solver "$solver" \
        --algorithm_type "$ALGO" \
        --skip_type "$SKIP" \
        --NFE "$nfe" \
        --CFG "$cfg" \
        --order "$ORDER" \
        --data "$DATA" \
        --save_root "$SAVE_ROOT" \
        --n_samples "$N_SAMPLES" \
        --seed_offset "$SEED_OFFSET" \
        --batch_size "$BATCH_SIZE" \
        --output_noise \
        --output_traj
        
    done
  done
done