#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import argparse
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

GMFLOW = os.path.join("submodules", "GMFlow")
sys.path.insert(0, GMFLOW)

# ===============================
# CLI: 요청대로 세 가지만 제어
# ===============================
def get_args():
    p = argparse.ArgumentParser(description="BNS training (only 3 overrides)")
    p.add_argument('--n_steps',    type=int, default=3)
    p.add_argument('--log_dir',    type=str, default=None, help="Override TensorBoard/log save dir")
    return p.parse_args()

args = get_args()

# ===============================
# Config (원문 유지 + 3가지만 덮어쓰기)
# ===============================
config = EasyDict()
config.backbone      = 'GMDiT'
config.batch_size    = 10
config.n_valid       = 100
config.CFG           = 1.4
config.latent_size   = (4, 32, 32)

# LR & Scheduler
config.base_lr       = 2e-3
config.end_lr        = 1e-4
config.total_steps   = 50*100        # 전체 학습 스텝

# ---- 여기만 CLI로 덮어씀 ----
config.n_steps       = args.n_steps
config.log_dir       = args.log_dir or config.log_dir
config.train_pt_dir  = '/dataset/gmdit/train1.4_1k'
config.valid_pt_dir  = '/dataset/gmdit/valid1.4_100'
# -----------------------------

# Loss
config.classifier = EasyDict()
config.losses = ['mse_loss']
config.main_loss = 'mse_loss'

os.makedirs(config.log_dir, exist_ok=True)

# ===============================
# Model (frozen)
# ===============================
from backbones.gmdit import GMDiT

if config.backbone == 'GMDiT':
    model = GMDiT(trainable=True)  # 내부 구현에 맞춰 유지
    model.set_freeze()
device = model.device
print(model)
print('done')

# ===============================
# Solver / Optimizer / Scheduler
# ===============================
from solvers.competing.bns.bns_solver import BNS_Solver

noise_schedule = model.get_noise_schedule()
solver = BNS_Solver(noise_schedule,
        config.n_steps,
        skip_type='time_uniform_flow',
        flow_shift=1.0,
        algorithm_type='vector_prediction',
        checkpoint=True).to(device)
optimizer = torch.optim.AdamW(solver.parameters(), lr=config.base_lr)

# ---- Scheduler: Pure Cosine ----
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config.total_steps,   # 전체 스텝에 걸쳐 한 번의 코사인
    eta_min=config.end_lr
)

print('solver/optimizer')

# ===============================
# Synthetic Dataset (drop-in)
# ===============================
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    """
    파일 I/O 없이 noise/target/cond를 즉석 생성.
    - noise  ~ N(0,1)
    - sample = zeros (시간 측정 안정적)
    - cond   = None  (필요시 dict로 확장 가능)
    """
    def __init__(self, length: int, latent_chw=(4, 32, 32), seed: int = 0):
        self.length = int(length)
        self.C, self.H, self.W = latent_chw
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noise = torch.randn((self.C, self.H, self.W), generator=self.g)
        target = torch.zeros((self.C, self.H, self.W))
        cond = torch.randint(1000, size=(1,))
        return {'noise': noise, 'sample': target, 'cond': cond}

# ===============================
# Dataset / Dataloader
# ===============================
train_dataset = SyntheticDataset(1000, config.latent_size)
print('len(train_dataset) :', len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

valid_dataset = SyntheticDataset(100, config.latent_size)
print('len(valid_dataset) :', len(valid_dataset))
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

print('dataloaders ready')


# ===============================
# Utils
# ===============================
def abort_if_bad(tag, value, step=None):
    v = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if (not math.isfinite(v)) or (v >= 100.0):
        msg = f"[EARLY-STOP] {tag} loss={v:.6f}" + (f" @ step {step}" if step is not None else "")
        print(msg, flush=True)
        raise RuntimeError(msg)

def save_checkpoint(global_step, save_dir, solver, optimizer):
    ckpt = {
        "global_step": int(global_step),
        "optim_state_dict": optimizer.state_dict(),
        "solver_state_dict": solver.state_dict(),
        "config": dict(config),
    }
    os.makedirs(save_dir, exist_ok=True)
    step_path = os.path.join(save_dir, f"step_{global_step:08d}.pt")
    torch.save(ckpt, step_path)
    return step_path

@torch.no_grad()
def get_valid_loss(valid_loader, device, solver):
    solver.eval()
    losses = []
    for batch in valid_loader:
        noises  = batch['noise'].to(device, non_blocking=True)
        conds   = batch['cond']
        targets = batch['sample'].to(device, non_blocking=True)
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.no_grad():
            latent_pred = solver.sample(noises, model_fn)['samples']
            loss = torch.log(F.mse_loss(latent_pred, targets))
            losses.append(loss.item())
    return np.mean(losses)

def _trimmed_mean_std_excl_minmax(values):
    """
    Return (mean, std) excluding a single min and max.
    - If len(values) == 0: returns (nan, nan)
    - If len(values) <= 2: fallback to simple mean & std over all values
    - Otherwise: drop one min and one max, then compute mean & std over the core
    Std is population std (ddof=0).
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")

    def _mean_std(arr):
        m = float(sum(arr)) / len(arr)
        if len(arr) <= 1:
            return m, 0.0
        var = sum((x - m) ** 2 for x in arr) / len(arr)  # population variance
        return m, var ** 0.5

    if n <= 2:
        return _mean_std(values)

    s = sorted(values)
    core = s[1:-1]  # exclude one min and one max
    return _mean_std(core)

import time
def do_train_loop(device, train_loader, solver, optimizer, global_step):
    solver.train()
    pbar = tqdm(train_loader)
    
    elapsed_times = {'sampling': [], 'backward': []}
    for batch in pbar:
        if global_step >= config.total_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        noises  = batch['noise'].to(device)
        conds   = batch['cond']
        targets = batch['sample'].to(device) 
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            torch.cuda.synchronize()
            t0 = time.time()
            latent_pred = solver.sample(noises, model_fn)['samples']
            torch.cuda.synchronize()
            elapsed_times['sampling'].append(time.time() - t0)
            torch.cuda.synchronize()
            t0 = time.time()
            loss = torch.log(F.mse_loss(latent_pred, targets))
                
        abort_if_bad("train", loss, global_step)  # ← 즉시 중단
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed_times['backward'].append(time.time() - t0)

        scheduler.step()   # ← lr 업데이트 포인트
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({'loss': loss.item(), 'lr': lr_now})
        global_step += 1

     # ---- 여기서 트리밍 평균/표준편차 출력 ----
    samp_mean, samp_std = _trimmed_mean_std_excl_minmax(elapsed_times['sampling'])  # ddof=0(모집단)
    bwd_mean,  bwd_std  = _trimmed_mean_std_excl_minmax(elapsed_times['backward'])
    n_iter = len(elapsed_times['sampling'])

    # 콘솔 출력 (ms)
    print(
        "[TIME] "
        f"sampling (excl min/max): {samp_mean*1000:.2f} ± {samp_std*1000:.2f} ms | "
        f"backward: {bwd_mean*1000:.2f} ± {bwd_std*1000:.2f} ms | "
        f"iters: {n_iter}"
    )
    
    return global_step


# ===============================
# Train
# ===============================
def main():
    writer = SummaryWriter(config.log_dir)
    print('tensorboard:', config.log_dir)

    global_step = 0
    while True:
        loss = get_valid_loss(valid_loader, device, solver)
        writer.add_scalar('valid_loss', loss, global_step)
        save_checkpoint(global_step, config.log_dir, solver, optimizer) 
        if global_step >= config.total_steps:
            break
        global_step = do_train_loop(device, train_loader, solver, optimizer, global_step)
        
    print('E-N-D')
    
if __name__ == "__main__":
    main()
