#!/usr/bin/env bash
#

DEVICES='0,1,2,3'

SAVE_ROOTS=(
  "/samplings/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE5)(CFG1.2)(ORDER2)"
  "/samplings/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG1.2)(ORDER2)"

)

for save_root in "${SAVE_ROOTS[@]}"; do
  echo "▶ Running --sample_path=$save_root"
  CUDA_VISIBLE_DEVICES=$DEVICES python -m runs.compute_clipscore calc --images="$save_root" \
    --prompts-npz prompts/mscoco2014_val.npz \
    --num 10000
done