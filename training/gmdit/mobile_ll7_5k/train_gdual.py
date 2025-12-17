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

# (arch, weight_fullname) — 문자열 그대로 Classifier에 넘겨 사용
CLASSIFIER_MODELS = [
  ("mobilenet_v3_large","MobileNet_V3_Large_Weights.IMAGENET1K_V2"),           # Rank: 84,  Acc@5: 92.566, GFLOPS: 0.22, FID : 9.44
]

# ===============================
# CLI: 요청대로 세 가지만 제어
# ===============================
def get_args():
    p = argparse.ArgumentParser(description="GDual training (only 3 overrides)")
    p.add_argument('--n_steps',    type=int, default=3)
    p.add_argument('--gamma_init',    type=float, default=0.0)
    p.add_argument('--tau_init',    type=float, default=0.0)
    p.add_argument('--n_classifiers',    type=int, default=1)
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
config.total_steps   = 5*1000        # 전체 학습 스텝

# ---- 여기만 CLI로 덮어씀 ----
config.n_steps       = args.n_steps
config.gamma_init    = args.gamma_init
config.tau_init      = args.tau_init
config.log_dir       = args.log_dir or config.log_dir
config.n_classifiers = args.n_classifiers
# -----------------------------

# Loss
config.losses = ['classifier']
config.main_loss = 'classifier'

os.makedirs(config.log_dir, exist_ok=True)

# ===============================
# Model (frozen)
# ===============================
from backbones.gmdit import GMDiT
from utils.general_classifier import Classifier

if config.backbone == 'GMDiT':
    model = GMDiT(trainable=True)  # 내부 구현에 맞춰 유지
    model.set_freeze()
device = model.device
print(model)

if 'classifier' in config.losses:
    classifiers = []
    for arch, weights in CLASSIFIER_MODELS[:config.n_classifiers]:
        classifier = Classifier(
                arch=arch,
                weights=weights,
            ).to(device)
        classifiers.append(classifier)
    
print('done')

# ===============================
# Solver / Optimizer / Scheduler
# ===============================
from solvers.taylor.solver.gdual_solver import GDual_Solver
from solvers.taylor.transform.loglinear_transform7 import LogLinearTransform
from solvers.taylor.extractor.table_extractor import Extractor

noise_schedule = model.get_noise_schedule()
extractor = Extractor(steps=config.n_steps)
transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
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

# ---- Resume (if latest pt exists) ----
from utils.util import get_latest_pt
resume_step = 0
latest = get_latest_pt(config.log_dir)
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


def get_classifier_loss(raw_output, targets):
    loss_list = []
    for classifier in classifiers:
        loss = classifier(raw_output, targets=targets)['loss']
        loss_list.append(loss)
    return torch.mean(torch.stack(loss_list))

@torch.no_grad()
def get_valid_loss(valid_noises, valid_conds, device, solver):
    solver.eval()
    losses = []
    start = 0
    while True:
        if start >= len(valid_noises):
            break
        noises = valid_noises[start:min(start+config.batch_size, len(valid_noises))]
        conds = valid_conds[start:min(start+config.batch_size, len(valid_noises))]
        start += config.batch_size
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latent_pred = solver.sample(noises, model_fn)['samples']
            if 'classifier' in config.losses:
                outputs = model.decode_vae(latent_pred, raw_output=True)
                loss = get_classifier_loss(outputs['raw_output'], targets=conds)
                losses.append(loss.item())

    return np.mean(losses)


def do_train_loop(device, writer, solver, optimizer, global_step):
    solver.train()
    pbar = tqdm(range(1000))
    
    for _, batch in enumerate(pbar):
        if global_step >= config.total_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        if config.main_loss == 'classifier':
            noises = torch.randn(config.batch_size, *config.latent_size).to(device, non_blocking=True)
            conds = torch.randint(0, 1000, size=(len(noises),)).to(device, non_blocking=True)
        
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latent_pred = solver.sample(noises, model_fn)['samples']
            if 'classifier' in config.main_loss:
                outputs = model.decode_vae(latent_pred, raw_output=True)
                loss = get_classifier_loss(outputs['raw_output'], targets=conds)
                
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

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()

    writer = SummaryWriter(config.log_dir)
    print('tensorboard:', config.log_dir)

    # global_step을 resume 지점부터 시작
    global_step = resume_step
    
    valid_noises = torch.randn(config.n_valid, *config.latent_size).to(device, non_blocking=True)
    valid_conds = torch.randint(0, 1000, size=(len(valid_noises),)).to(device, non_blocking=True)
    while True:
        loss = get_valid_loss(valid_noises, valid_conds, device, solver)
        writer.add_scalar('valid_loss', loss, global_step)
        save_checkpoint(global_step, config.log_dir, solver, optimizer) 

        if global_step >= config.total_steps:
            break
        
        global_step = do_train_loop(device, writer, solver, optimizer, global_step)

    print('E-N-D')
    
if __name__ == "__main__":
    main()
