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
    p = argparse.ArgumentParser(description="GDual training (only 3 overrides)")
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
config.total_steps   = 20*1000        # 전체 학습 스텝

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

from utils.fid import FIDInception
inception = FIDInception().to(device)

# ===============================
# Solver / Optimizer / Scheduler
# ===============================
from solvers.taylor.solver.gdual_solver import GDual_Solver
from solvers.taylor.transform.logaffine_transform import LogAffineTransform
from solvers.taylor.extractor.table_extractor import Extractor

noise_schedule = model.get_noise_schedule()
extractor = Extractor(steps=config.n_steps)
transform = LogAffineTransform(gamma_push=True, gamma_max=2, tau_offset=1, kappa_max=2, eps=1e-2)
solver = GDual_Solver(
    noise_schedule,
    steps=config.n_steps,
    transform=transform,
    param_extractor=extractor,
    skip_type="time_uniform_flow",
    flow_shift=1.0,
    pred_order=1,
    corr_order=2,
    order1_kappa=True,
    order2_kappa=True,
    use_corrector=True,
    time_learning=True,
    train_mode=True,
    checkpoint=True
).to(device)

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
# Dataset / Dataloader
# ===============================
from datasets.pt_dataset import PtDataset

train_dataset = PtDataset(config.train_pt_dir)
print('len(train_dataset) :', len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

valid_dataset = PtDataset(config.valid_pt_dir)
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
            decoded1 = model.decode_vae(latent_pred, raw_output=True)
            inception_features1 = inception.encode(decoded1['raw_outputs'])

            decoded2 = model.decode_vae(targets, raw_output=True)
            inception_features2 = inception.encode(decoded2['raw_outputs'])
            loss = F.mse_loss(inception_features1, inception_features2)
            losses.append(loss.item())
    return np.mean(losses)

def do_train_loop(device, train_loader, solver, optimizer, global_step):
    solver.train()
    pbar = tqdm(train_loader)
    
    for batch in pbar:
        if global_step >= config.total_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        noises  = batch['noise'].to(device, non_blocking=True)
        conds   = batch['cond']
        targets = batch['sample'].to(device, non_blocking=True)        
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latent_pred = solver.sample(noises, model_fn)['samples']
            decoded1 = model.decode_vae(latent_pred, raw_output=True)
            inception_features1 = inception.encode(decoded1['raw_outputs'])

            decoded2 = model.decode_vae(targets, raw_output=True)
            inception_features2 = inception.encode(decoded2['raw_outputs'])
            loss = F.mse_loss(inception_features1, inception_features2)
                
        abort_if_bad("train", loss, global_step)  # ← 즉시 중단
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        optimizer.step()
        scheduler.step()   # ← lr 업데이트 포인트
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({'loss': loss.item(), 'lr': lr_now})
        global_step += 1
        
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
