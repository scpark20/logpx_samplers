#!/usr/bin/env bash
set -euo pipefail

# ==== GPU 설정 (쉼표로 구분) ====
# 예) "0,1,2,3" 또는 "0"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

# nproc 계산
IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#_GPU_IDS[@]}

# 포트(충돌 방지용 임의값). 필요시 외부에서 MASTER_PORT 지정 가능.
MASTER_PORT="${MASTER_PORT:-29501}"

# ==== 공통 하이퍼파라미터 ====
TAG=dit_euler50k
SAVE_ROOT=samplings/dit/euler50k
MODEL=DiT
DATA=ImageNet
BATCH_SIZE=10          # per-rank batch size (GPU당 배치)
ALGO=data_prediction
SKIP=time_uniform
ORDER=1
N_SAMPLES=50000
SEED_OFFSET=0

# ==== 스윕 설정 ====
SOLVERS=("Euler")
NFES=(3 5 7 9)
CFGS=(4.0)

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      echo "▶ torchrun (nproc=${NPROC}, GPUs=${CUDA_VISIBLE_DEVICES}) | ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg}"

      torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
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