#!/usr/bin/env bash
set -euo pipefail

# GPU 수동 지정 (여러 개면 쉼표로)
export CUDA_VISIBLE_DEVICES=0,1

# 🔇 BLAS 스레드 수 제한
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

TAG=mobile_interpolation1
MODEL=DiT
DATA=ImageNet
BATCH_SIZE=10          # GPU당 배치
ALGO=dual_prediction
SKIP=time_uniform
FLOW_SHIFT=1.0
ORDER=2
N_SAMPLES=50000
SEED_OFFSET=0

SOLVERS=("Dual-Solver")

# (왼쪽 NFE, 오른쪽 NFE, 중간 NFE)
NFES=(
  "3 5 4"
  "5 7 6"
  "7 9 8"

  "3 6 4"
  "3 6 5"
  "6 9 7"
  "6 9 8"

  "3 9 4"
  "3 9 5"
  "3 9 6"
  "3 9 7"
  "3 9 8"
)

CFGS=(1.5)

# 사용 GPU 개수 -> nproc
IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#_GPU_IDS[@]}

BASE_OUT="samplings"   # SAVE_ROOT 베이스

for solver in "${SOLVERS[@]}"; do
  for triple in "${NFES[@]}"; do
    # triple = "3 5 4" 같은 문자열 -> 세 변수로 분리
    read -r NFE_L NFE_R NFE_M <<< "$triple"

    for cfg in "${CFGS[@]}"; do
      # 저장 디렉토리에 좌/우/중간 NFE 표시
      SAVE_ROOT="${BASE_OUT}/${MODEL}/cfg${cfg}/L${NFE_L}_R${NFE_R}_M${NFE_M}/${solver}/${N_SAMPLES}"

      # 해당 조합으로 만들어 둔 pt 파일 디렉터리
      # (python 쪽에서 logs/dit/mobile_interpolation1/s${steps[0]}${steps[1]}${steps[2]} 에 저장했으므로)
      PT_DIR="logs/dit/mobile_interpolation1/s${NFE_L}${NFE_R}${NFE_M}"

      echo "▶ torchrun (nproc=${NPROC}, GPUs=${CUDA_VISIBLE_DEVICES}) | ${MODEL} | solver=${solver} | NFE=${NFE_M} (from ${NFE_L}→${NFE_R}) | CFG=${cfg}"
      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
      DIST_BACKEND=gloo torchrun --standalone --nproc_per_node="${NPROC}" \
        -m runs.sample_ddp_gloo \
          --tag "$TAG" \
          --model "$MODEL" \
          --solver "$solver" \
          --algorithm_type "$ALGO" \
          --skip_type "$SKIP" \
          --flow_shift "$FLOW_SHIFT" \
          --NFE "$NFE_M" \
          --CFG "$cfg" \
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
