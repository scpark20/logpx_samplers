#!/usr/bin/env bash
set -e

# 여기서 GPU 번호 수동 지정 (여러 개면 쉼표로)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 🔇 torchrun OMP 배너 억제(사전에 지정)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

TAG=rn_clip
MODEL=SANA
DATA=MSCOCO2014_valid_30k
BATCH_SIZE=5          # GPU당 배치
ALGO=dual_prediction
SKIP=time_uniform_flow
FLOW_SHIFT=3.0
ORDER=2
N_SAMPLES=30000
SEED_OFFSET=0

SOLVERS=("Dual-Solver")
NFES=(3 4 5 6)
CFGS=(4.5)

# 사용 GPU 개수 -> nproc
IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#_GPU_IDS[@]}

BASE_OUT="samplings"   # SAVE_ROOT의 베이스

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
        SAVE_ROOT="${BASE_OUT}/${MODEL}/${cfg}/${nfe}/${solver}/${N_SAMPLES}/${TAG}"
        PT_DIR="logs/sana/rn/s${nfe}"
        echo "▶ torchrun (nproc=${NPROC}, GPUs=${CUDA_VISIBLE_DEVICES}) | ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg}"
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
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
            --order "$ORDER" \
            --data "$DATA" \
            --save_root "$SAVE_ROOT" \
            --pt_dir "$PT_DIR" \
            --pt_criterion "latest" \
            --n_samples "$N_SAMPLES" \
            --seed_offset "$SEED_OFFSET" \
            --batch_size "$BATCH_SIZE" \
            --output_clip_score \
            --clip_model "ViT-L/14, ViT-L/14@336px"
      done
    done
  done
done