#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, math, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import trange

import torch
import torch.distributed as dist
from diffusers import DiTPipeline, DPMSolverMultistepScheduler


def parse_args():
    p = argparse.ArgumentParser(description="DiT-XL/2 ImageNet256 → DPMSolver(10) → PNG(+NPZ) [DDP]")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=50, help="per-GPU batch size")
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--build_npz", action="store_true", help="샘플링 후 arr_0 NPZ 생성(메모리 큼)")
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


def build_npz_from_pngs(png_dir: Path, out_npz: Path, n: int):
    arr = np.empty((n, 256, 256, 3), dtype=np.uint8)  # ~9.8GB
    for i in trange(n, desc="Building NPZ"):
        im = Image.open(png_dir / f"{i:06d}.png").convert("RGB")
        arr[i] = np.asarray(im, dtype=np.uint8)
    np.savez_compressed(out_npz, arr_0=arr)


def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    png_dir = out_dir / "samples_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    local_rank, rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cc_major = torch.cuda.get_device_capability()[0] if device.type == "cuda" else 0
    torch_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16

    pipe = DiTPipeline.from_pretrained(a.model_id, torch_dtype=torch_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=(rank != 0))

    labels = labels_imagenet_balanced(a.n_samples, seed=a.seed)

    g_start, g_end = split_range(a.n_samples, world_size, rank)
    for i in trange(g_start, g_end, a.batch_size, disable=(rank != 0), desc=f"Rank{rank}"):
        j = min(g_end, i + a.batch_size)
        idxs = list(range(i, j))
        batch_labels = labels[i:j].tolist()
        gens = [torch.Generator(device=device).manual_seed(a.seed + gi) for gi in idxs]

        with torch.inference_mode():
            out = pipe(
                class_labels=batch_labels,
                num_inference_steps=a.steps,
                guidance_scale=a.cfg,
                generator=gens,       # per-sample 시드 → 월드사이즈와 무관한 재현성
                output_type="pil",
            )
        for gi, img in zip(idxs, out.images):
            img.save(png_dir / f"{gi:06d}.png", format="PNG")

    barrier()

    if rank == 0 and a.build_npz:
        build_npz_from_pngs(png_dir, out_dir / "samples_arr0_256x256x3_uint8.npz", a.n_samples)

    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
