import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_h_14, ViT_H_14_Weights,
)

class ViTClassifier(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 arch="vit_b_16",
                 weights=ViT_B_16_Weights.IMAGENET1K_V1,
                 freeze_backbone=True,
                 label_smoothing=0.0,
                 net_dtype=torch.bfloat16,
                 image_size="auto"):
        super().__init__()
        builder = {"vit_b_16": vit_b_16, "vit_l_16": vit_l_16, "vit_h_14": vit_h_14}[arch]
        m = builder(weights=weights)  # 공식 가중치 로드
        # 헤드 교체(필요 시)
        head = getattr(getattr(m, "heads", None), "head", None)
        if head is not None and head.out_features != num_classes:
            m.heads.head = nn.Linear(head.in_features, num_classes)
        elif hasattr(m, "classifier") and m.classifier.out_features != num_classes:
            m.classifier = nn.Linear(m.classifier.in_features, num_classes)

        self.m = m.to(dtype=net_dtype)
        self.m.eval() if freeze_backbone else self.m.train()
        for p in self.m.parameters(): p.requires_grad_(not freeze_backbone)

        meta = getattr(weights, "meta", {}) if hasattr(weights, "meta") else {}
        mean = meta.get("mean", (0.485, 0.456, 0.406))
        std  = meta.get("std",  (0.229, 0.224, 0.225))

        # ---- target size 결정 ----
        if image_size == "auto":
            _, H, W = meta.get("input_size", (3, 224, 224))
        elif isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.target_hw = (int(H), int(W))
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
        self.label_smoothing = float(label_smoothing)

    def forward(self, x, targets=None, input_range="-1..1", clamp="ste"):
        p = next(self.m.parameters()); x = x.to(p.device)
        if x.shape[1] == 1: x = x.repeat(1,3,1,1)
        if input_range == "-1..1": x = (x + 1) * 0.5
        if clamp == "hard": x = x.clamp(0,1)
        elif clamp == "ste": x = x + (x.clamp(0,1) - x).detach()
        if x.shape[-2:] != self.target_hw:
            x = F.interpolate(x.float(), self.target_hw, mode="bilinear",
                              align_corners=False, antialias=True)
        else:
            x = x.float()
        x = (x - self.mean) / self.std
        x = x.to(p.dtype)
        logits = self.m(x)  # torchvision ViT는 Tensor 반환
        out = {"logits": logits, "pred": logits.argmax(1)}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits.float(), targets.long(),
                                          label_smoothing=self.label_smoothing)
        return out

# 사용 예
# model = ViTClassifier(arch="vit_b_16", weights=ViT_B_16_Weights.IMAGENET1K_V1)
# model_h = ViTClassifier(arch="vit_h_14", weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
