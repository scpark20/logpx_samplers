#!/usr/bin/env bash
#
# SANA 모델: Euler / DPM-Solver
# NFE = 5,6,8,10
# order = 2
# n_samples = 10000
# CFG = 1.5,3.5,5.5,7.5,9.5
#
set -e

MODEL=SANA
DATA=MSCOCO2017
SAVE_ROOT=samplings/
BATCH_SIZE=5        # 필요에 따라 조정
ALGO=data_prediction
SKIP=time_uniform_flow
FLOW_SHIFT=3.0
ORDER=2
N_SAMPLES=1000

SOLVERS=("Euler" "DPM-Solver" "UniPC")
NFES=(5 6 8 10)
CFGS=(1.5 3.5 5.5 7.5 9.5)

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      echo "▶ Running ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg}"
      python -m runs.sample \
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
        --batch_size "$BATCH_SIZE"
    done
  done
done