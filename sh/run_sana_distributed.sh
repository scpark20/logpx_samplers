#!/usr/bin/env bash
#
# DEVICES='0,1,2,3' GPU 4개 사용 
# SANA 모델: Euler / DPM-Solver
# NFE = 5,6,8,10
# order = 2
# n_samples = 10000
# CFG = 1.5,3.5,5.5,7.5,9.5
#
set -e

DEVICES='0,1,2,3,4,5,6,7'

# MODEL=DiT
# DATA=Imagenet
# SKIP=time_uniform
# ORDER=2
# FLOW_SHIFT=1.0

MODEL=SANA
DATA=MSCOCO2017
SKIP=time_uniform_flow
ORDER=2
FLOW_SHIFT=3.0

SAVE_ROOT=samplings/

BATCH_SIZE=10        # 필요에 따라 조정
ALGO=data_prediction
N_SAMPLES=10000
RESULT_TYPE="all"

# SOLVERS=("Euler" "DPM-Solver" "UniPC")
# SOLVERS=("Euler")
# CFGS=(1.5 3.5 5.5 7.5 9.5)
SOLVERS=("DPM-Solver")
NFES=(10 20)
CFGS=(1.5 3.5)

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      echo "▶ Running ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg}"
      CUDA_VISIBLE_DEVICES=$DEVICES python -m runs.sample_distributed \
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
        --batch_size "$BATCH_SIZE" \
        --result_type "$RESULT_TYPE"
    done
  done
done