#!/usr/bin/env bash
set -euo pipefail

# 여기서 GPU 번호 수동 지정 (여러 개면 쉼표로)
export CUDA_VISIBLE_DEVICES=0,1

# 🔇 torchrun OMP 배너 억제
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

TAG=mobile_ll6_5k
MODEL=GMDiT
DATA=ImageNet
BATCH_SIZE=50          # GPU당 배치
ALGO=dual_prediction
SKIP=time_uniform_flow
FLOW_SHIFT=1.0
ORDER=2
N_SAMPLES=50000
SEED_OFFSET=0

SOLVERS=("Dual-Solver_LL6")
NFES=(9)
CFGS=(1.4)
GAMMAS=(0.0 1.0)
# (tx, te) 쌍을 문자열 튜플로 직접 정의
TAUS=(
  "-2.0 2.0"
  "2.0 -2.0"
)


# 사용 GPU 개수 -> nproc
IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#_GPU_IDS[@]}

BASE_OUT="samplings"

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      for g in "${GAMMAS[@]}"; do
        for pair in "${TAUS[@]}"; do

          # tx, te 분리
          read -r tx te <<< "$pair"

          SAVE_ROOT="${BASE_OUT}/${MODEL}/${cfg}/${nfe}/${solver}/${N_SAMPLES}/g${g}_tx${tx}_te${te}"
          PT_DIR="logs/gmdit/mobile_ll6_5k/s${nfe}_g${g}_tx${tx}_te${te}"

          echo "▶ torchrun | GPUs=${CUDA_VISIBLE_DEVICES} | ${MODEL} | ${solver} | NFE=${nfe} | CFG=${cfg} | g=${g} | tx=${tx} | te=${te}"

          DIST_BACKEND=gloo torchrun --standalone --nproc_per_node="${NPROC}" \
            -m runs.sample_ddp_gloo \
              --tag "$TAG" \
              --model "$MODEL" \
              --solver "$solver" \
              --algorithm_type "$ALGO" \
              --skip_type "$SKIP" \
              --flow_shift "$FLOW_SHIFT" \
              --NFE "$nfe" \
              --CFG "$cfg" \
              --gamma_init "$g" \
              --tau_x_init "$tx" \
              --tau_e_init "$te" \
              --order "$ORDER" \
              --data "$DATA" \
              --save_root "$SAVE_ROOT" \
              --pt_dir "$PT_DIR" \
              --pt_criterion "latest" \
              --n_samples "$N_SAMPLES" \
              --seed_offset "$SEED_OFFSET" \
              --batch_size "$BATCH_SIZE" \
              --output_inception

        done
      done
    done
  done
done
