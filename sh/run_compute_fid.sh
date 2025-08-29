#!/usr/bin/env bash
#

# REF_STATS_PATH=ref_stats/imagenet-256x256.npz
REF_STATS_PATH=ref_stats/mscoco2014_val_10k.npz


SAVE_ROOTS=(
  "/samplings/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE5)(CFG1.2)(ORDER2)"
  "/samplings/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG1.2)(ORDER2)"
)

N_SAMPLES=10000

for save_root in "${SAVE_ROOTS[@]}"; do

  echo "▶ Running --ref_path=$REF_STATS_PATH --sample_path=$save_root --n_samples=$N_SAMPLES"
  python -m runs.compute_fid \
    --ref_path "$REF_STATS_PATH" \
    --n_samples "$N_SAMPLES" \
    --sample_path "$save_root"

done