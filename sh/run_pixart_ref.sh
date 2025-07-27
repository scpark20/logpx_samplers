#!/usr/bin/env bash
set -e

TAG=pixartref
MODEL=PixArt-Sigma
DATA=MSCOCO2017
SAVE_ROOT=samplings/pixart/ref
BATCH_SIZE=5
ALGO=data_prediction
SKIP=time_uniform
ORDER=2
N_SAMPLES=100

SOLVERS=("Euler")
NFES=(200)
CFGS=(1.5 3.5 5.5 7.5)

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
        --batch_size "$BATCH_SIZE"
    done
  done
done
