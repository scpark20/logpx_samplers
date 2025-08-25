import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm  # get_model

class Classifier(nn.Module):
    """
    Inference-only classifier (no head replacement, no training paths).
    - 항상 eval, 모든 파라미터 requires_grad=False (입력 gradient는 유지)
    - ViTClassifier 스타일 전처리: bilinear resize(크롭 없음) + (x-mean)/std
    - weights는 그대로 get_model에 전달 (Enum / "IMAGENET1K_V1" / "DEFAULT" / None)
    - inception_v3: pretrained 사용 시 aux_logits=True 자동 설정(가중치 검증 통과)
    """
    def __init__(
        self,
        arch: str = "vit_b_16",
        weights=None,                    # e.g., "IMAGENET1K_V1" or Weights Enum or None
        net_dtype=torch.float32,
        image_size="auto",               # "auto" -> (299 if inception_v3 else 224); int or (H,W)도 허용
        input_range: str = "-1..1",      # "-1..1" | "0..1" | "0..255"
        clamp: str = "ste",              # "ste" | "hard" | None
        model_kwargs: dict = None,
    ):
        super().__init__()
        self.arch = arch
        self.input_range = input_range
        self.clamp = clamp
        self.net_dtype = net_dtype

        # Inception v3는 pretrained일 때 aux_logits=True를 요구함
        extra = dict(model_kwargs or {})
        if arch == "inception_v3" and weights is not None:
            extra.setdefault("aux_logits", True)

        # 모델 생성 (헤드 교체 없음)
        m = tvm.get_model(arch, weights=weights, **extra)
        self.m = m.to(dtype=net_dtype)

        # 항상 eval + 파라미터 동결
        self._lock_eval_and_freeze()

        # mean/std 및 타깃 크기 설정 (weights가 Enum이면 meta 사용, 아니면 기본값)
        if hasattr(weights, "meta"):
            meta = weights.meta
            mean = meta.get("mean", (0.485, 0.456, 0.406))
            std  = meta.get("std",  (0.229, 0.224, 0.225))
            if image_size == "auto":
                _, H, W = meta.get("input_size", (3, 224, 224))
            elif isinstance(image_size, int):
                H = W = image_size
            else:
                H, W = image_size
        else:
            mean = (0.485, 0.456, 0.406)
            std  = (0.229, 0.224, 0.225)
            if image_size == "auto":
                H = W = 299 if arch == "inception_v3" else 224
            elif isinstance(image_size, int):
                H = W = image_size
            else:
                H, W = image_size

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1))
        self.target_hw = (int(H), int(W))

    def _lock_eval_and_freeze(self):
        super().train(False)
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    # 사용자가 .train(True) 호출해도 무시하고 eval 유지
    def train(self, mode: bool = True):
        super().train(False)
        if hasattr(self, "m"):
            self.m.eval()
        return self

    def _normalize_input_range(self, x: torch.Tensor):
        # x: (B,C,H,W) or (C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:   # grayscale -> RGB
            x = x.repeat(1, 3, 1, 1)

        if self.input_range == "-1..1":
            x = (x + 1.0) * 0.5
        elif self.input_range == "0..255":
            x = x / 255.0

        if self.clamp == "hard":
            x = x.clamp(0, 1)
        elif self.clamp == "ste":
            x = x + (x.clamp(0, 1) - x).detach()
        return x

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        # 전처리
        p = next(self.m.parameters())
        device = p.device

        x = self._normalize_input_range(x).to(device=device, dtype=torch.float32)
        if x.shape[-2:] != self.target_hw:
            x = F.interpolate(x, self.target_hw, mode="bilinear",
                              align_corners=False, antialias=True)
        x = (x - self.mean.to(device)) / self.std.to(device)
        x = x.to(dtype=p.dtype)

        # 모델 추론
        logits = self.m(x)
        # InceptionOutputs 같은 namedtuple 처리
        if not isinstance(logits, torch.Tensor):
            if hasattr(logits, "logits"):
                logits = logits.logits
            else:
                logits = logits[0]

        out = {"logits": logits, "pred": logits.argmax(1)}
        if targets is not None:
            # 학습은 아니지만, 평가/지도 신호용 CE 계산은 제공
            out["loss"] = F.cross_entropy(logits.float(), targets.long())
        return out
