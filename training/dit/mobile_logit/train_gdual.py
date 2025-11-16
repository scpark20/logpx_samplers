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

# (arch, weight_fullname) — 문자열 그대로 Classifier에 넘겨 사용
CLASSIFIER_MODELS = [
  ("mobilenet_v3_large","MobileNet_V3_Large_Weights.IMAGENET1K_V2"),           # Rank: 84,  Acc@5: 92.566, GFLOPS: 0.22, FID : 9.44
]

# ===============================
# CLI: 요청대로 세 가지만 제어
# ===============================
def get_args():
    p = argparse.ArgumentParser(description="Gdual training (only 3 overrides)")
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
config.n_valid       = 100
config.CFG           = 1.5
config.latent_size   = (4, 32, 32)

# LR & Scheduler
config.base_lr       = 2e-3
config.end_lr        = 1e-4
config.total_steps   = 5*1000        # 전체 학습 스텝

# ---- 여기만 CLI로 덮어씀 ----
config.n_steps       = args.n_steps
config.log_dir       = args.log_dir
config.train_pt_dir  = '/dataset/dit/train1.5_1k'
config.valid_pt_dir  = '/dataset/dit/valid1.5_100'
# -----------------------------

os.makedirs(config.log_dir, exist_ok=True)

# ===============================
# Model (frozen)
# ===============================
from backbones.dit import DiT
from utils.general_classifier import Classifier

if config.backbone == 'DiT':
    model = DiT(trainable=True)  # 내부 구현에 맞춰 유지
    model.set_freeze()
device = model.device
print(model)

arch = 'mobilenet_v3_large'
weights = 'MobileNet_V3_Large_Weights.IMAGENET1K_V2'
classifier = Classifier(arch=arch, weights=weights).to(device)
for p in classifier.parameters():
    p.requires_grad_(False)
classifier.eval()

print(classifier)
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
# Dataset / Dataloader
# ===============================
from datasets.pt_dataset import PtDataset

train_dataset = PtDataset(config.train_pt_dir)
print('len(train_dataset) :', len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)


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

def cross_entropy_between_logits(logits_s, logits_t, T=1.0):
    # logits_s, logits_t: (batch, num_classes)
    # T: temperature (지식증류 할 때 자주 씀, 필요 없으면 1.0)
    
    # student 쪽: log_softmax
    log_q = F.log_softmax(logits_s / T, dim=-1)  # log q(c)
    # target 쪽: softmax
    p = F.softmax(logits_t / T, dim=-1)          # p(c)

    # H(p, q) = - sum_c p(c) log q(c)
    ce = -(p * log_q).sum(dim=-1).mean()

    # distillation에서는 보통 T**2 를 곱해줌 (옵션)
    return ce * (T * T)

def LOGIT_LOSS(pred, trgt, decoder):
    logit_pred = classifier(decoder(pred))['logits']
    logit_trgt = classifier(decoder(trgt))['logits']
    loss = cross_entropy_between_logits(logit_pred, logit_trgt)
    return loss

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
            loss = LOGIT_LOSS(latent_pred, targets, lambda x: model.decode_vae(x, raw_output=True)['raw_output'])
            loss = loss.mean()
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
            loss = LOGIT_LOSS(latent_pred, targets, lambda x: model.decode_vae(x, raw_output=True)['raw_output'])
            loss = loss.mean()

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
