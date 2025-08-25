#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn.functional as F
from torchvision import models as tvm  # get_weight 용

# ------------------------------------------------------------
# 20개 모델 (arch, weights_fullname) — 그대로 사용
# ------------------------------------------------------------
MODELS_FOR_EXP = [
  ("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1"),
  ("mobilenet_v3_small", "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
  ("mnasnet0_75",        "MNASNet0_75_Weights.IMAGENET1K_V1"),
  ("mobilenet_v2",       "MobileNet_V2_Weights.IMAGENET1K_V1"),
  ("efficientnet_b0",    "EfficientNet_B0_Weights.IMAGENET1K_V1"),
  ("regnet_y_400mf",     "RegNet_Y_400MF_Weights.IMAGENET1K_V1"),
  ("alexnet",            "AlexNet_Weights.IMAGENET1K_V1"),
  ("regnet_x_800mf",     "RegNet_X_800MF_Weights.IMAGENET1K_V1"),
  ("efficientnet_b2",    "EfficientNet_B2_Weights.IMAGENET1K_V1"),
  ("densenet121",        "DenseNet121_Weights.IMAGENET1K_V1"),
  ("resnet50",           "ResNet50_Weights.IMAGENET1K_V1"),
  ("vit_b_32",           "ViT_B_32_Weights.IMAGENET1K_V1"),
  ("swin_t",             "Swin_T_Weights.IMAGENET1K_V1"),
  ("maxvit_t",           "MaxVit_T_Weights.IMAGENET1K_V1"),
  ("inception_v3",       "Inception_V3_Weights.IMAGENET1K_V1"),  # 포함 요구
  ("convnext_small",     "ConvNeXt_Small_Weights.IMAGENET1K_V1"),
  ("vit_b_16",           "ViT_B_16_Weights.IMAGENET1K_V1"),      # 포함 요구
  ("efficientnet_v2_m",  "EfficientNet_V2_M_Weights.IMAGENET1K_V1"),
  ("regnet_y_32gf",      "RegNet_Y_32GF_Weights.IMAGENET1K_V1"),
  ("vit_l_16",           "ViT_L_16_Weights.IMAGENET1K_V1"),
]

def main():
  p = argparse.ArgumentParser(description="One-shot CE loss check over 20 classifiers")
  p.add_argument("--batch_size", type=int, default=10)
  p.add_argument("--n_steps",    type=int, default=5)
  p.add_argument("--CFG",        type=float, default=4.0)
  p.add_argument("--latent_h",   type=int, default=32)
  p.add_argument("--latent_w",   type=int, default=32)
  p.add_argument("--seed",       type=int, default=0)
  args = p.parse_args()

  # ---------------- basic setup ----------------
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  latent_size = (4, args.latent_h, args.latent_w)  # (C,H,W)

  # ---------------- DiT backbone (frozen) ----------------
  from backbones.dit import DiT
  dit = DiT(trainable=True)
  dit.set_freeze()
  noise_schedule = dit.get_noise_schedule()

  # ---------------- Solver (sampling once) ----------------
  from solvers.taylor.solver.gdual_solver import GDual_Solver
  from solvers.taylor.transform.logaffine_transform import LogAffineTransform
  from solvers.taylor.extractor.table_extractor import Extractor

  extractor = Extractor(steps=args.n_steps)
  transform = LogAffineTransform(gamma_push=True, gamma_max=2, tau_offset=1, kappa_max=2, eps=1e-2)
  solver = GDual_Solver(
      noise_schedule,
      steps=args.n_steps,
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

  # ---------------- make one sample ----------------
  # (same noises/conds for all classifiers)
  noises = torch.randn(args.batch_size, *latent_size, device=device)
  # 분류 타깃은 간단히 랜덤 클래스(0..999)
  class_ids = torch.randint(0, 1000, (args.batch_size,), device=device).long()

  model_fn = dit.get_model_fn(noise_schedule, pos_conds=class_ids, guidance_scale=args.CFG)
  with torch.no_grad():
    latent_pred = solver.sample(noises, model_fn)
    # pixel [0,1] 범위로 디코드 (ViTClassifier 스타일 전처리와 호환)
    sample_pred = dit.decode_vae(latent_pred, raw_output=True).clamp(0, 1)

  # ---------------- use YOUR Classifier (exactly as you wrote) ----------------
  # 경로는 네 프로젝트에 맞춰 수정해
  from utils.general_classifier import Classifier  # ← 네가 앞서 만든 inference-only 버전

  print(f"[device] {device}  |  batch={args.batch_size}, steps={args.n_steps}, CFG={args.CFG}")
  print("----- one-shot CE loss per classifier -----")

  for idx, (arch, weight_fullname) in enumerate(MODELS_FOR_EXP):
    # 가중치는 풀네임 문자열 그대로 전달 (Enum으로 변환하고 싶다면 tvm.get_weight 사용)
    try:
      # Enum으로 정확한 meta(input_size/mean/std)를 쓰고 싶으면 아래 라인으로 대체:
      # weights = tvm.get_weight(weight_fullname)
      weights = weight_fullname

      clf = Classifier(
        arch=arch,
        weights=weights,
        net_dtype=torch.float32,         # 안전하게 FP32
        # Inception은 네 Classifier가 aux_logits 자동 True 처리
      ).to(device)

      with torch.no_grad():
        out = clf(sample_pred, targets=class_ids)
        loss = float(out["loss"].detach().cpu())
        print(f"[{idx:02d}] {arch:18s} | {weight_fullname:45s} | CE: {loss:.6f}")

      # 메모리 정리 (GPU)
      del clf
      if device.startswith("cuda"):
        torch.cuda.empty_cache()

    except Exception as e:
      print(f"[{idx:02d}] {arch:18s} | ERROR: {e}")

  print("----- done -----")

if __name__ == "__main__":
  main()
