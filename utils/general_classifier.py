import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from torchvision.models._api import WeightsEnum

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

        # ---- helper: coerce string/DEFAULT to real Weights Enum so .meta works
        weights = self._coerce_weights(arch, weights)

        # Inception v3 needs aux_logits=True when using pretrained
        extra = dict(model_kwargs or {})
        if arch == "inception_v3" and weights is not None:
            extra.setdefault("aux_logits", True)

        # Build model
        m = tvm.get_model(arch, weights=weights, **extra).to(dtype=net_dtype)
        self.m = m
        self._lock_eval_and_freeze()

        # Resolve mean/std from weights.meta if available
        meta = getattr(weights, "meta", {}) if weights is not None else {}
        mean = meta.get("mean", (0.485, 0.456, 0.406))
        std  = meta.get("std",  (0.229, 0.224, 0.225))

        # Resolve target size
        if image_size == "auto":
            if hasattr(m, "image_size") and m.image_size is not None:
                H = W = int(m.image_size)            # e.g., vit_h_14 SWAG -> 518
            else:
                _, H, W = meta.get("input_size", (3, 299 if arch == "inception_v3" else 224, 299 if arch == "inception_v3" else 224))
        elif isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1))
        self.target_hw = (int(H), int(W))

    # --- helper ---
    @staticmethod
    def _coerce_weights(arch: str, weights):
        """
        Accepts Enum/String/None:
        - "DEFAULT" -> weights enum DEFAULT
        - "IMAGENET1K_SWAG_E2E_V1" (etc.) -> corresponding enum
        - Enum/None -> returned as-is
        """
        if isinstance(weights, WeightsEnum) or weights is None:
            return weights
        if isinstance(weights, str):
            try:
                enum_cls = tvm.get_model_weights(arch)
                if weights.upper() == "DEFAULT":
                    return enum_cls.DEFAULT
                # attribute name must match exactly
                if hasattr(enum_cls, weights):
                    return getattr(enum_cls, weights)
            except Exception:
                pass
        return weights

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
