#!/usr/bin/env bash
set -e

# 여기서 GPU 번호 수동 지정
CUDA_VISIBLE_DEVICES=0

TAG=train1.5_1k
MODEL=DiT
DATA=ImageNet
SAVE_ROOT=samplings/dit/train1.5_1k
BATCH_SIZE=10
ALGO=data_prediction
SKIP=time_uniform
ORDER=1
N_SAMPLES=1000
SEED_OFFSET=1

SOLVERS=("Euler")
NFES=(200)
CFGS=(1.5)

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      echo "▶ Running ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg} | GPU=${CUDA_VISIBLE_DEVICES}"
      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
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
        --output_noise
    done
  done
done
