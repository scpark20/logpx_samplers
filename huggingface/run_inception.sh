# 사용 GPU 지정(예: 4장)
export CUDA_VISIBLE_DEVICES=0,1

# 50k PNG + NPZ 생성
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m huggingface.simple_dit256_dpms10_ddp_inception \
  --out_dir samplings/dit256_dpms10_inception --n_samples 50000 \
  --steps 10 --batch_size 500 --cfg 1.5
