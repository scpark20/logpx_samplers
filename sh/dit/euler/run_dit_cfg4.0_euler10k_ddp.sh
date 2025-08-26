#!/usr/bin/env bash
set -e

# 여기서 GPU 번호 수동 지정 (여러 개면 쉼표로)
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=0,1

TAG=dit_euler10k
SAVE_ROOT=samplings/dit/euler10k
MODEL=DiT
DATA=ImageNet
BATCH_SIZE=50          # GPU당 배치
ALGO=data_prediction
SKIP=time_uniform
ORDER=1
N_SAMPLES=10000
SEED_OFFSET=0

SOLVERS=("Euler")
NFES=(3 5 7 9)
CFGS=(4.0)

# 사용 GPU 개수 -> nproc
IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#_GPU_IDS[@]}

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      echo "▶ torchrun (nproc=${NPROC}, GPUs=${CUDA_VISIBLE_DEVICES}) | ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg}"
      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
      torchrun --standalone --nproc_per_node="${NPROC}" \
        -m runs.sample_ddp \
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
          --inception
    done
  done
done
