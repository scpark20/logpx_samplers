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
config.backbone      = 'DiT'
config.batch_size    = 10
config.CFG           = 1.5
config.val_every     = 100
config.latent_size   = (4, 32, 32)

# LR & Scheduler
config.base_lr       = 2e-3
config.end_lr        = 1e-4
config.total_steps   = 10*1000+1        # 전체 학습 스텝

# ---- 여기만 CLI로 덮어씀 ----
config.n_steps       = args.n_steps
config.log_dir       = args.log_dir or config.log_dir
# -----------------------------

# Loss
config.classifier = EasyDict()
config.losses = ['classifier']
config.main_loss = 'classifier'

os.makedirs(config.log_dir, exist_ok=True)

# ===============================
# Model (frozen)
# ===============================
from backbones.dit import DiT
from utils.vit import ViTClassifier

if config.backbone == 'DiT':
    model = DiT(trainable=True)  # 내부 구현에 맞춰 유지
    model.set_freeze()
device = model.device
print(model)
if 'classifier' in config.losses:
    classifier = ViTClassifier().to(device)
print('done')

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
    skip_type="time_uniform",
    pred_order=1,
    corr_order=2,
    order1_kappa=True,
    order2_kappa=True,
    use_corrector=True,
    time_learning=True,
    train_mode=True
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

from IPython.display import clear_output

def do_train_loop(device, writer, solver, optimizer, global_step):
    solver.train()
    pbar = tqdm(range(100))
    losses = []
    for _, batch in enumerate(pbar):
        if global_step >= config.total_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        if config.main_loss == 'classifier':
            noises = torch.randn(config.batch_size, *config.latent_size).to(device, non_blocking=True)
            conds = torch.randint(0, 1000, size=(len(noises),))
        else:
            noises = batch['noise'].to(device, non_blocking=True)
            conds  = batch['cond']
            targets= batch['sample'].to(device, non_blocking=True)
            target_features = batch['inception_feature'][:, 0].to(device, non_blocking=True)

        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latent_pred = solver.sample(noises, model_fn)
            if 'classifier' == config.main_loss:
                outputs = model.decode_vae(latent_pred, raw_output=True)
                class_ids = conds.to(device, non_blocking=True).long()
                loss = classifier(outputs['raw_output'], targets=class_ids)["loss"]
                
        abort_if_bad("train", loss, global_step)  # ← 즉시 중단
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        optimizer.step()
        scheduler.step()   # ← lr 업데이트 포인트
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({'loss': loss.item(), 'lr': lr_now})
        global_step += 1
        
        clear_output()
        
    if global_step >= config.total_steps:
        return global_step

    writer.add_scalar('train_loss', np.mean(losses), global_step)
    save_checkpoint(global_step, config.log_dir, solver, optimizer)

    return global_step


# ===============================
# Train
# ===============================
def main():
    writer = SummaryWriter(config.log_dir)
    print('tensorboard:', config.log_dir)

    global_step = 0
    while True:
        if global_step >= config.total_steps:
            break
        global_step = do_train_loop(device, writer, solver, optimizer, global_step)
    print('E-N-D')
    
if __name__ == "__main__":
    main()
