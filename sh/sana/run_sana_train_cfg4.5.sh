#!/usr/bin/env bash
set -e

# 여기서 GPU 번호 수동 지정
CUDA_VISIBLE_DEVICES=0

TAG=sana_train4.5
MODEL=SANA
DATA=MSCOCO2017
SAVE_ROOT=samplings/sana/train4.5
BATCH_SIZE=10
ALGO=data_prediction
SKIP=time_uniform_flow
FLOW_SHIFT=3.0
ORDER=1
N_SAMPLES=10000
SEED_OFFSET=1 # train : 1, valid : 0

SOLVERS=("Euler")
NFES=(200)
CFGS=(4.5)

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
        --flow_shift "$FLOW_SHIFT" \
        --NFE "$nfe" \
        --CFG "$cfg" \
        --order "$ORDER" \
        --data "$DATA" \
        --save_root "$SAVE_ROOT" \
        --n_samples "$N_SAMPLES" \
        --seed_offset "$SEED_OFFSET" \
        --batch_size "$BATCH_SIZE" \
        --output_noise \
        --inception \
        --clip
    done
  done
done
