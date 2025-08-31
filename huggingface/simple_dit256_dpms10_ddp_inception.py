#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, math, argparse
from pathlib import Path
from tqdm import trange

import numpy as np
import torch
import torch.distributed as dist
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from utils.fid import FIDInception  # <- FID feature encoder (pytorch-fid wrapper)


def parse_args():
    p = argparse.ArgumentParser(description="DiT-XL/2 ImageNet256 → DPMSolver(10) → Inception features (.pt) [DDP]")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=50, help="per-GPU batch size")
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_id", type=str, default="facebook/DiT-XL-2-256")
    return p.parse_args()


def ddp_setup():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank, dist.get_rank(), dist.get_world_size()
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 0, 1


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def labels_imagenet_balanced(n, seed=0):
    reps = (n + 999) // 1000
    labels = np.tile(np.arange(1000, dtype=np.int64), reps)[:n]
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return labels


def split_range(n, world_size, rank):
    per = (n + world_size - 1) // world_size
    s = per * rank
    e = min(n, s + per)
    return s, e


def compact(x: torch.Tensor) -> torch.Tensor:
    """Move to CPU float32 & contiguous for safe saving."""
    return x.detach().to("cpu", dtype=torch.float32).contiguous()


def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    save_dir = out_dir  # 바로 out_dir에 저장 (원하면 / "samples_pt" 등으로 바꿔도 됨)
    save_dir.mkdir(parents=True, exist_ok=True)

    local_rank, rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if device.type == "cuda":
        cc_major = torch.cuda.get_device_capability()[0]
        torch_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
    else:
        torch_dtype = torch.float32  # CPU fallback은 fp32

    # --- Model / scheduler ---
    pipe = DiTPipeline.from_pretrained(a.model_id, torch_dtype=torch_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=(rank != 0))

    # --- Inception encoder (keep in fp32 for stability) ---
    inception = FIDInception(dims=2048, net_dtype=torch.float32).to(device)  # .eval() already set in class

    # --- Build class labels & ranges ---
    labels = labels_imagenet_balanced(a.n_samples, seed=a.seed)
    g_start, g_end = split_range(a.n_samples, world_size, rank)

    for i in trange(g_start, g_end, a.batch_size, disable=(rank != 0), desc=f"Rank{rank}"):
        j = min(g_end, i + a.batch_size)
        idxs = list(range(i, j))
        batch_labels = labels[i:j].tolist()
        gens = [torch.Generator(device=device).manual_seed(a.seed + gi) for gi in idxs]

        with torch.inference_mode():
            # 1) sample PIL images
            out = pipe(
                class_labels=batch_labels,
                num_inference_steps=a.steps,
                guidance_scale=a.cfg,
                generator=gens,         # per-sample seed (global-index based)
                output_type="pil",      # we need PIL for FIDInception.forward
            )
            # 2) encode to 2048-d Inception features (Tensor shape: [B, 2048])
            feats = inception(out.images).detach()

        # 3) save per-sample .pt (one file per image)
        for gi, lbl, feat in zip(idxs, batch_labels, feats):
            gseed = int(a.seed + gi)
            output = {
                "index": int(gi),
                "label": int(lbl),
                "seed": gseed,
                "inception_feature": compact(feat),  # CPU float32
            }
            torch.save(output, save_dir / f"{gi:06d}.pt")

    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
