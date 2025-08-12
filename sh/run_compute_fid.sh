#!/usr/bin/env bash
#


# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS1.0)(NFE250)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS1.0)(NFE100)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS1.0)(NFE50)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(DPM-Solver)(data_prediction)(logSNR)(FS1.0)(NFE250)(CFG1.5)(ORDER1)/"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(DPM-Solver)(data_prediction)(time_uniform)(FS1.0)(NFE250)(CFG1.5)(ORDER2)"
SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/DiT(Imagenet)(Euler)(data_prediction)(time_uniform)(FS1.0)(NFE250)(CFG1.5)(ORDER2)"
REF_STATS_PATH=ref_stats/imagenet-256x256.npz



# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG4.0)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE20)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE20)(CFG3.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE50)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE50)(CFG3.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE250)(CFG1.5)(ORDER2)"
# SAVE_ROOT="/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE250)(CFG3.5)(ORDER2)"
# REF_STATS_PATH=ref_stats/ms_coco-512x512.npz


N_SAMPLES=1000

# for solver in "${SOLVERS[@]}"; do

  echo "▶ Running ${MODEL} | path=${SAVE_ROOT}"
  echo "ARGS: --ref_path=$REF_STATS_PATH --n_samples=$N_SAMPLES --sample_path=$SAVE_ROOT"
  python -m runs.compute_fid \
    --ref_path "$REF_STATS_PATH" \
    --n_samples "$N_SAMPLES" \
    --sample_path "$SAVE_ROOT"

# done