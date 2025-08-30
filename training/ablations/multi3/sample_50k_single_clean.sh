#!/usr/bin/env bash
set -euo pipefail

# ===== GPU 선택 =====
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===== BLAS/OMP 노이즈 억제 =====
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

TAG=multi3
MODEL=DiT
DATA=ImageNet
BATCH_SIZE=50
ALGO=dual_prediction
SKIP=time_uniform
ORDER=2
N_SAMPLES=50000
SEED_OFFSET=0

SOLVERS=("Dual-Solver")
NFES=(9)
CFGS=(1.5)

BASE_OUT="samplings"

# 단일 프로세스 강제 (torchrun 잔여 환경변수 무시)
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK NODE_RANK

for solver in "${SOLVERS[@]}"; do
  for cfg in "${CFGS[@]}"; do
    for nfe in "${NFES[@]}"; do
      SAVE_ROOT="${BASE_OUT}/${MODEL}/ablations/multi3/cfg${cfg}_s${nfe}_N${N_SAMPLES}"
      PT_DIR="logs/ablations/multi3/s${nfe}"

      echo "▶ python(single) | GPU=${CUDA_VISIBLE_DEVICES} | ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg}"

      python -u -m runs.sample_ddp_gloo \
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
        --pt_dir "$PT_DIR" \
        --pt_criterion "latest" \
        --n_samples "$N_SAMPLES" \
        --seed_offset "$SEED_OFFSET" \
        --batch_size "$BATCH_SIZE" \
        --output_inception \
        --output_clean_inception

      # 만약 내가 준 single 전용 파이썬 러너(runs.sample_single)를 쓴다면 위 줄 대신:
      # python -u -m runs.sample_single  \
      #   (인자 동일)
    done
  done
done
