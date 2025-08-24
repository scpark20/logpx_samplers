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
    p.add_argument('--except_gamma', action='store_true',)
    p.add_argument('--except_tau', action='store_true',)
    p.add_argument('--except_kappa', action='store_true',)
    p.add_argument('--shared_taukappa', action='store_true',)
    return p.parse_args()

args = get_args()

# 하나만 선택(또는 0개) 강제
only_one = int(args.except_gamma) + int(args.except_tau) + int(args.except_kappa) + int(args.shared_taukappa)
if only_one > 1:
    raise ValueError("Choose at most one of: --except_gamma, --except_tau, --except_kappa, --shared_taukappa")

# 기본값
DEFAULTS = dict(
    backbone='DiT',
    valid_pt_dir='/dataset/dit/eval4.0',
    batch_size=10,
    CFG=4.0,
    val_every=100,
    latent_size=(4,32,32),
    base_lr=2e-3,
    end_lr=1e-4,
    total_steps=10_001,
    n_steps=3,
    log_dir='logs/ablations/pred_corr/tmp',
    losses=['inception','PSNR','classifier'],
    main_loss='classifier',
    except_gamma=False,
    except_tau=False,
    except_kappa=False,
    shared_taukappa=False,
)

# DEFAULTS → CLI 덮어쓰기(None은 무시)
config = EasyDict(DEFAULTS)
config.update({k: v for k, v in vars(args).items() if v is not None})

os.makedirs(config.log_dir, exist_ok=True)

# ===============================
# Model (frozen)
# ===============================
from backbones.dit import DiT
from utils.inception import FIDInception
from utils.vit import ViTClassifier

if config.backbone == 'DiT':
    model = DiT(trainable=True)  # 내부 구현에 맞춰 유지
    model.set_freeze()
device = model.device
print(model)
inception = FIDInception().to(device)
if 'classifier' in config.losses:
    classifier = ViTClassifier().to(device)
print('done')

# ===============================
# Solver / Optimizer / Scheduler
# ===============================
from solvers.taylor.solver.gdual_solver import GDual_Solver
from solvers.taylor.transform.ablation_logaffine_transform import LogAffineTransform
from solvers.taylor.extractor.table_extractor import Extractor

noise_schedule = model.get_noise_schedule()
extractor = Extractor()
transform = LogAffineTransform(
    gamma_push=True, gamma_max=2, tau_offset=1, kappa_max=2, eps=1e-2,
    except_gamma=config.except_gamma,
    except_tau=config.except_tau,
    except_kappa=config.except_kappa,
    shared_taukappa=config.shared_taukappa)
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
# Dataset / Dataloader
# ===============================
from datasets.pt_dataset import PtDataset

if config.main_loss != 'classifier':
    train_dataset = PtDataset(config.train_pt_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    print('len(train_dataset) :', len(train_dataset))
    
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
def get_valid_loss(device, solver):
    solver.eval()
    losses = {}
    if 'PSNR' in config.losses:
        losses['PSNR'] = []
    if 'inception' in config.losses:
        losses['inception'] = []
    if 'classifier' in config.losses:
        losses['classifier'] = []
    
    for batch in valid_loader:
        noises = batch['noise'].to(device, non_blocking=True)
        conds  = batch['cond']
        targets= batch['sample'].to(device, non_blocking=True)
        target_features= batch['inception_feature'][:, 0].to(device, non_blocking=True)
        
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.no_grad():
            latent_pred = solver.sample(noises, model_fn)
            if 'PSNR' in losses:
                psnr_loss = torch.log(F.mse_loss(latent_pred, targets) + 1e-8)
                losses['PSNR'].append(psnr_loss.item())

            if 'inception' in config.losses or 'classifier' in config.losses:
                sample_pred = model.decode_vae(latent_pred, raw_output=True)
    
                if 'inception' in config.losses:
                    pred = inception(sample_pred)
                    inception_loss = F.mse_loss(pred, target_features)
                    losses['inception'].append(inception_loss.item())

                if 'classifier' in config.losses:
                    class_ids = conds.to(device, non_blocking=True).long()
                    ce_loss = classifier(sample_pred, targets=class_ids)["loss"]
                    losses['classifier'].append(ce_loss.item())

    for key in losses:
        losses[key] = float(np.mean(losses[key]))
    return losses

from IPython.display import clear_output

def do_train_loop(device, writer, solver, optimizer, global_step):
    solver.train()
    if config.main_loss == 'classifier':
        pbar = tqdm(range(100))
    else:
        pbar = tqdm(train_loader)
        
    for _, batch in enumerate(pbar):
        if global_step >= config.total_steps:
            break

        #if global_step > 0 and global_step % config.val_every == 0:
        if global_step % config.val_every == 0:
            valid_losses = get_valid_loss(device, solver)
            for key in valid_losses:
                writer.add_scalar(key, valid_losses[key], global_step)
            save_checkpoint(global_step, config.log_dir, solver, optimizer)

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
            if 'PSNR' == config.main_loss:
                loss = torch.log(F.mse_loss(latent_pred, targets) + 1e-8)
                
            if 'inception' == config.main_loss or 'classifier' == config.main_loss:
                sample_pred = model.decode_vae(latent_pred, raw_output=True)
    
                if 'inception' == config.main_loss:
                    pred = inception(sample_pred)
                    loss = F.mse_loss(pred, target_features)
                    
                if 'classifier' == config.main_loss:
                    class_ids = conds.to(device, non_blocking=True).long()
                    loss = classifier(sample_pred, targets=class_ids)["loss"]
                    
        abort_if_bad("train", loss, global_step)  # ← 즉시 중단

        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        optimizer.step()
        scheduler.step()   # ← lr 업데이트 포인트
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({'loss': loss.item(), 'lr': lr_now})
        global_step += 1
        
        clear_output()

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
