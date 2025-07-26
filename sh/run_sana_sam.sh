#!/usr/bin/env bash
set -e

TAG=sana
MODEL=SANA
DATA=MSCOCO2017
SAVE_ROOT=samplings/sana/sam
BATCH_SIZE=5
ALGO=data_prediction
SKIP=time_uniform_flow
ORDER=2
N_SAMPLES=5

SOLVERS=("Euler" "DPM-Solver")
NFES=(5 10)
CFGS=(3.5 5.5)
FLOW_SHIFTS=(1.0 3.0)

for solver in "${SOLVERS[@]}"; do
  for nfe in "${NFES[@]}"; do
    for cfg in "${CFGS[@]}"; do
      for flow_shift in "${FLOW_SHIFTS[@]}"; do
        echo "▶ Running ${MODEL} | solver=${solver} | NFE=${nfe} | CFG=${cfg} | flow_shift=${flow_shift}"
        python -m runs.sample \
          --tag "$TAG" \
          --model "$MODEL" \
          --solver "$solver" \
          --algorithm_type "$ALGO" \
          --skip_type "$SKIP" \
          --flow_shift "$flow_shift" \
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
done
