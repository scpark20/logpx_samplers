import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm  # get_model, get_model_weights, get_weight, list_models


# ------------------------- utils: weights resolver -------------------------
def _resolve_weights(arch: str, weights):
    """
    weights:
      - Enum 인스턴스: 그대로
      - "EnumClass.KEY" 문자열: tvm.get_weight(...)로 해석
      - "DEFAULT"/"IMAGENET1K_V2" 등 키 문자열: tvm.get_model_weights(arch)에서 조회
      - None: 무가중치
    반환:
      - Enum 인스턴스 or None
    """
    # Enum-like 객체(Weight enum 인스턴스)면 그대로
    if hasattr(weights, "meta") and hasattr(weights, "transforms"):
        return weights

    if isinstance(weights, str):
        # "ViT_B_16_Weights.IMAGENET1K_V1" 같은 풀네임
        if "." in weights:
            try:
                return tvm.get_weight(weights)
            except Exception:
                pass
        # 모델별 enum class에서 조회
        try:
            enum_cls = tvm.get_model_weights(arch)  # torchvision >= 0.14
            if hasattr(enum_cls, weights):
                return getattr(enum_cls, weights)
            if weights.upper() == "DEFAULT" and hasattr(enum_cls, "DEFAULT"):
                return enum_cls.DEFAULT
        except Exception:
            # 없는 모델/버전이면 None로 두고 진행
            return None
        return None
    # None or 기타 타입은 그대로
    return weights


# ---------------------------- head replacement ----------------------------
def _replace_classifier_head(m: nn.Module, num_classes: int):
    """
    torchvision 분류 모델들의 최종 분류층을 num_classes에 맞게 교체.
    가능한 경로: heads.head (ViT), fc (ResNet), classifier(여러 모델), 마지막 Linear 스캔.
    """
    # ViT류
    if hasattr(m, "heads") and hasattr(m.heads, "head") and isinstance(m.heads.head, nn.Linear):
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, num_classes)
        return m.heads.head

    # ResNet류
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m.fc

    # EfficientNet/ConvNeXt/MobileNet/VGG/DenseNet 등
    if hasattr(m, "classifier"):
        head = getattr(m, "classifier")
        if isinstance(head, nn.Linear):
            in_f = head.in_features
            m.classifier = nn.Linear(in_f, num_classes)
            return m.classifier
        if isinstance(head, nn.Sequential):
            # 뒤에서부터 Linear 찾기
            for idx in range(len(head) - 1, -1, -1):
                if isinstance(head[idx], nn.Linear):
                    in_f = head[idx].in_features
                    head[idx] = nn.Linear(in_f, num_classes)
                    return head[idx]

    # 마지막 Linear를 스캔해 교체 (최후 수단)
    last_linear_name, last_linear = None, None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            last_linear_name, last_linear = name, mod
    if last_linear is not None:
        parent = m
        parts = last_linear_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        in_f = last_linear.in_features
        setattr(parent, parts[-1], nn.Linear(in_f, num_classes))
        return getattr(parent, parts[-1])

    return None


# ------------------------------- Classifier --------------------------------
class Classifier(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 arch: str = "resnet50",          # ex) "inception_v3", "vit_b_16", "swin_t", ...
                 weights=None,                     # "DEFAULT" | "IMAGENET1K_V2" | "ViT_B_16_Weights.IMAGENET1K_V1" | Enum | None
                 freeze_backbone: bool = True,
                 tune_last_layer: bool = False,    # freeze 시 마지막 head만 학습
                 label_smoothing: float = 0.0,
                 net_dtype=torch.bfloat16,
                 use_weight_transforms: bool = True,
                 image_size="auto",                # use_weight_transforms=False일 때만 사용
                 input_range: str = "-1..1",       # "-1..1" | "0..1" | "0..255"
                 clamp: str = "ste",               # "ste" | "hard" | None
                 model_kwargs: dict = None):       # 특정 모델 전용 추가 인자 (예: {"quantize": True})
        super().__init__()
        self.arch = arch
        self.label_smoothing = float(label_smoothing)
        self.net_dtype = net_dtype
        self.input_range = input_range
        self.clamp = clamp
        self.use_weight_transforms = bool(use_weight_transforms)
        self.target_hw = None

        # weights 해석
        self.weights_obj = _resolve_weights(arch, weights)

        # Inception 특례(aux_logits=False) 등 모델별 kwargs 합치기
        extra = dict(model_kwargs or {})
        if arch == "inception_v3":
            # 가중치가 있으면 torchvision이 aux_logits=True를 요구함(verify 통과용).
            # eval()에서는 aux 분기 계산 안 하므로 오버헤드 없음.
            extra.setdefault("aux_logits", self.weights_obj is not None)

        # 문자열 이름으로 모델 인스턴스화
        self.m = tvm.get_model(arch, weights=self.weights_obj, **extra)

        # 분류 헤드 교체(요청 시)
        self.head = None
        if num_classes is not None:
            self.head = _replace_classifier_head(self.m, num_classes)

        # dtype 적용
        self.m = self.m.to(dtype=net_dtype)

        # 학습/동결 설정
        if freeze_backbone:
            self.m.eval()
            for p in self.m.parameters():
                p.requires_grad_(False)
            if tune_last_layer and self.head is not None:
                for p in self.head.parameters():
                    p.requires_grad_(True)
                self.head.train()
        else:
            self.m.train()
            for p in self.m.parameters():
                p.requires_grad_(True)

        # 전처리 설정
        if self.use_weight_transforms and hasattr(self.weights_obj, "transforms"):
            self.transforms = self.weights_obj.transforms()
            meta = getattr(self.weights_obj, "meta", {}) if self.weights_obj is not None else {}
            self.target_hw = tuple(meta.get("input_size", (3, 224, 224))[1:])
        else:
            # 수동 경로: meta에서 mean/std/size 가져오되 없으면 기본값
            meta = getattr(self.weights_obj, "meta", {}) if hasattr(self.weights_obj, "meta") else {}
            mean = meta.get("mean", (0.485, 0.456, 0.406))
            std  = meta.get("std",  (0.229, 0.224, 0.225))
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))
            if image_size == "auto":
                _, H, W = meta.get("input_size", (3, 224, 224))
            elif isinstance(image_size, int):
                H = W = image_size
            else:
                H, W = image_size
            self.target_hw = (int(H), int(W))
            self.transforms = None  # 수동 전처리 사용

    # -------------------------- preprocessing helpers --------------------------
    def _normalize_input_range(self, x: torch.Tensor):
        # x: (B,C,H,W) or (C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if self.input_range == "-1..1":
            x = (x + 1) * 0.5
        elif self.input_range == "0..255":
            x = x / 255.0

        if self.clamp == "hard":
            x = x.clamp(0, 1)
        elif self.clamp == "ste":
            x = x + (x.clamp(0, 1) - x).detach()
        return x

    def _preprocess_batch(self, x: torch.Tensor, device, out_dtype):
        x = self._normalize_input_range(x)
        x = x.to(device=device, dtype=torch.float32)

        if self.use_weight_transforms and self.transforms is not None:
            # weights.transforms()는 (C,H,W) 단위 입력을 기대 -> 배치에 루프 적용
            xs = []
            for i in range(x.shape[0]):
                xs.append(self.transforms(x[i]))  # (3,Ht,Wt)
            x = torch.stack(xs, dim=0)
            return x.to(device=device, dtype=out_dtype)

        # 수동 전처리: resize + normalize
        if x.shape[-2:] != self.target_hw:
            x = F.interpolate(x, self.target_hw, mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean.to(device)) / self.std.to(device)
        return x.to(device=device, dtype=out_dtype)

    # --------------------------------- forward ---------------------------------
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        p = next(self.m.parameters())
        device = p.device
        x = self._preprocess_batch(x, device=device, out_dtype=self.net_dtype)

        logits = self.m(x)
        # Inception 등에서 namedtuple일 수 있음 → logits 필드로 정규화
        if not isinstance(logits, torch.Tensor):
            if hasattr(logits, "logits"):
                logits = logits.logits
            else:
                logits = logits[0]

        out = {"logits": logits, "pred": logits.argmax(1)}
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.float(), targets.long(), label_smoothing=self.label_smoothing
            )
        return out


# ------------------------------- usage examples -------------------------------
# 1) 문자열로 아키텍처/가중치 지정 (권장)
# inception = Classifier(arch="inception_v3", weights="DEFAULT", freeze_backbone=True)
# r50      = Classifier(arch="resnet50",     weights="IMAGENET1K_V2")
# vitb     = Classifier(arch="vit_b_16",     weights="ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1")
# swin_t   = Classifier(arch="swin_t",       weights="DEFAULT")
# en_b4    = Classifier(arch="efficientnet_b4",   weights="IMAGENET1K_V1")
# cx_small = Classifier(arch="convnext_small",    weights="IMAGENET1K_V1")
# env2_m   = Classifier(arch="efficientnet_v2_m", weights="IMAGENET1K_V1")

# 2) 잘 모르는 이름을 넣었을 때 사용가능한 모델을 보고 싶다면:
# print(tvm.list_models())                         # 모든 모델
# print(tvm.list_models(module=tvm, include=["*v3*"]))  # 패턴 필터링
