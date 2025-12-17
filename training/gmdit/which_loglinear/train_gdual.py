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

CLASSIFIER_MODELS = [
  ("vit_h_14",          "ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1"),   # Rank: 1,   Acc@5: 98.694, GFLOPS: 1016.72
  ("regnet_y_128gf",    "RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),  # Rank: 6,   Acc@5: 97.844, GFLOPS: 127.52
  ("regnet_y_16gf",     "RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),   # Rank: 12,  Acc@5: 97.244, GFLOPS: 15.91
  ("convnext_base",     "ConvNeXt_Base_Weights.IMAGENET1K_V1"),                # Rank: 18,  Acc@5: 96.870, GFLOPS: 15.36
  ("efficientnet_b5",   "EfficientNet_B5_Weights.IMAGENET1K_V1"),              # Rank: 24,  Acc@5: 96.628, GFLOPS: 10.27
  ("regnet_y_16gf",     "RegNet_Y_16GF_Weights.IMAGENET1K_V2"),                # Rank: 30,  Acc@5: 96.328, GFLOPS: 15.91
  ("swin_v2_t",         "Swin_V2_T_Weights.IMAGENET1K_V1"),                    # Rank: 36,  Acc@5: 96.132, GFLOPS: 5.94
  ("swin_t",            "Swin_T_Weights.IMAGENET1K_V1"),                       # Rank: 42,  Acc@5: 95.776, GFLOPS: 4.49
  ("regnet_y_32gf",     "RegNet_Y_32GF_Weights.IMAGENET1K_V1"),                # Rank: 48,  Acc@5: 95.340, GFLOPS: 32.28
  ("regnet_y_8gf",      "RegNet_Y_8GF_Weights.IMAGENET1K_V1"),                 # Rank: 54,  Acc@5: 95.048, GFLOPS: 8.47
  ("regnet_y_3_2gf",    "RegNet_Y_3_2GF_Weights.IMAGENET1K_V1"),               # Rank: 60,  Acc@5: 94.576, GFLOPS: 3.18
  ("resnet152",         "ResNet152_Weights.IMAGENET1K_V1"),                    # Rank: 66,  Acc@5: 94.046, GFLOPS: 11.51
  ("resnet101",         "ResNet101_Weights.IMAGENET1K_V1"),                    # Rank: 72,  Acc@5: 93.546, GFLOPS: 7.8
  ("regnet_y_800mf",    "RegNet_Y_800MF_Weights.IMAGENET1K_V1"),               # Rank: 78,  Acc@5: 93.136, GFLOPS: 0.83
  ("mobilenet_v3_large","MobileNet_V3_Large_Weights.IMAGENET1K_V2"),           # Rank: 84,  Acc@5: 92.566, GFLOPS: 0.22
  ("regnet_y_400mf",    "RegNet_Y_400MF_Weights.IMAGENET1K_V1"),               # Rank: 90,  Acc@5: 91.716, GFLOPS: 0.40
  ("regnet_x_400mf",    "RegNet_X_400MF_Weights.IMAGENET1K_V1"),               # Rank: 96,  Acc@5: 90.950, GFLOPS: 0.41
  ("mobilenet_v2",      "MobileNet_V2_Weights.IMAGENET1K_V1"),                 # Rank: 102, Acc@5: 90.286, GFLOPS: 0.30
  ("shufflenet_v2_x1_0","ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"),           # Rank: 108, Acc@5: 88.316, GFLOPS: 0.14
  ("alexnet",           "AlexNet_Weights.IMAGENET1K_V1"),                      # Rank: 114, Acc@5: 79.066, GFLOPS: 0.71
]

# ===============================
# CLI: 요청대로 세 가지만 제어
# ===============================
def get_args():
    p = argparse.ArgumentParser(description="GDual training (only 3 overrides)")
    p.add_argument('--n_steps',    type=int, default=3)
    p.add_argument('--n_classifiers',    type=int, default=0)
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
    arch, weights = CLASSIFIER_MODELS[config.n_classifiers]
    classifier = Classifier(
            arch=arch,
            weights=weights,
        ).to(device)
    
print('done')

# ===============================
# Solver / Optimizer / Scheduler
# ===============================
from solvers.taylor.solver.gdual_solver import GDual_Solver
from solvers.taylor.transform.loglinear_transform import LogLinearTransform
from solvers.taylor.extractor.table_extractor import Extractor

noise_schedule = model.get_noise_schedule()
extractor = Extractor(steps=config.n_steps)
transform = LogLinearTransform(gamma_push=True, gamma_max=2, kappa_max=2, eps=1e-2)
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
    loss = classifier(raw_output, targets=targets)['loss']
    return loss

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

    global_step = 0
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
