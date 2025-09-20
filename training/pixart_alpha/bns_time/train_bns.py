#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
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

#from utils.util import get_latest_pt

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
config.backbone      = 'PixArt-Alpha'
config.batch_size    = 10
config.n_valid       = 100
config.CFG           = 3.5
config.latent_size   = (4, 64, 64)

# LR & Scheduler
config.base_lr       = 2e-3
config.end_lr        = 1e-4
config.total_steps   = 20*1000        # 전체 학습 스텝

# ---- 여기만 CLI로 덮어씀 ----
config.n_steps       = args.n_steps
config.log_dir       = args.log_dir or config.log_dir
config.train_pt_dir  = '/dataset/pixart_alpha3.5/train3.5_1k'
config.valid_pt_dir  = '/dataset/pixart_alpha3.5/valid3.5_100'
# -----------------------------

# Loss
config.classifier = EasyDict()
config.losses = ['mse_loss']
config.main_loss = 'mse_loss'

os.makedirs(config.log_dir, exist_ok=True)

# ===============================
# Model (frozen)
# ===============================
from backbones.pixart_alpha import PixArtAlpha

if config.backbone == 'PixArt-Alpha':
    model = PixArtAlpha(trainable=True)  # 내부 구현에 맞춰 유지
    model.set_freeze()
device = model.device
print(model)
print('done')

# ===============================
# Solver / Optimizer / Scheduler
# ===============================
from solvers.competing.bns.bns_solver_sep import BNS_Solver

noise_schedule = model.get_noise_schedule()
solver = BNS_Solver(noise_schedule,
        config.n_steps,
        skip_type='time_uniform',
        algorithm_type='dual_prediction',
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

# ---- Resume (if latest pt exists) ----
resume_step = 0
latest = None
if latest is not None:
    print(f"[RESUME] loading: {latest}")
    ckpt = torch.load(latest, map_location='cpu', weights_only=False)  # state dict은 장치 무관하게 로드 후 사용
    solver.load_state_dict(ckpt["solver_state_dict"])
    optimizer.load_state_dict(ckpt["optim_state_dict"])
    resume_step = int(ckpt.get("global_step", 0))

    # 스케줄러 상태가 저장되어 있으면 그대로 복구
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    else:
        # 없으면 현재 스텝에 맞춰 1회 동기화 (CosineAnnealingLR은 step(epoch) 지원)
        if resume_step > 0:
            scheduler.step(resume_step - 1)

    # 로드 결과 출력 (학습률 확인용)
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"[RESUME] global_step={resume_step}, lr={lr_now:.3e}")
else:
    print("[RESUME] no checkpoint found; starting from scratch")


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
        cond = 'cond'
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

def _trimmed_mean_excl_minmax(values):
    """Return mean excluding a single min and max. If len<=2, fallback to simple mean."""
    n = len(values)
    if n == 0:
        return float("nan")
    if n <= 2:
        return float(sum(values)) / n
    s = sorted(values)
    core = s[1:-1]
    return float(sum(core)) / len(core)

import time
def do_train_loop(device, train_loader, solver, optimizer, global_step):
    solver.train()
    pbar = tqdm(train_loader)
    
    elapsed_times = {'sampling': [], 'backward': []}
    for batch in pbar:
        if global_step >= config.total_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        noises  = batch['noise'].to(device, non_blocking=True)
        conds   = batch['cond']
        targets = batch['sample'].to(device, non_blocking=True)        
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            t0 = time.time()
            latent_pred = solver.sample(noises, model_fn)['samples']
            elapsed_times['sampling'].append(time.time() - t0)
            t0 = time.time()
            loss = torch.log(F.mse_loss(latent_pred, targets))
                
        abort_if_bad("train", loss, global_step)  # ← 즉시 중단
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        optimizer.step()
        elapsed_times['backward'].append(time.time() - t0)

        scheduler.step()   # ← lr 업데이트 포인트
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({'loss': loss.item(), 'lr': lr_now})
        global_step += 1

     # ---- 여기서 트리밍 평균 출력 ----
    samp_avg = _trimmed_mean_excl_minmax(elapsed_times['sampling'])
    bwd_avg  = _trimmed_mean_excl_minmax(elapsed_times['backward'])
    n_iter   = len(elapsed_times['sampling'])

    # 콘솔 출력 (ms)
    print(f"[TIME] sampling avg (excl min/max): {samp_avg*1000:.2f} ms | "
          f"backward avg (excl min/max): {bwd_avg*1000:.2f} ms | "
          f"iters: {n_iter}")
    
    return global_step


# ===============================
# Train
# ===============================
def main():
    writer = SummaryWriter(config.log_dir)
    print('tensorboard:', config.log_dir)

    # global_step을 resume 지점부터 시작
    global_step = resume_step

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
