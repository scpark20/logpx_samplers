import torch, torch.nn.functional as F
from torch import nn
from pytorch_fid.inception import InceptionV3
from torchvision.models import inception_v3, Inception_V3_Weights

class FIDInception(nn.Module):
    """No-crop resize → Inception(pool3). 입력 bf16 OK, 기본 입력범위 -1..1, 기본 clamp=STE."""
    def __init__(self, dims=2048, net_dtype=torch.bfloat16):
        super().__init__()
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).eval().to(dtype=net_dtype)
        for p in self.net.parameters(): p.requires_grad_(False)

    def forward(self, x, input_range="-1..1", clamp_mode="ste"):
        p = next(self.net.parameters()); x = x.to(p.device)
        if input_range == "-1..1": x = (x + 1) * 0.5  # -> [0,1] (이론상)
        # clamp 옵션
        if clamp_mode == "hard":
            x = x.clamp(0, 1)
        elif clamp_mode == "ste":
            xc = x.clamp(0, 1)
            x  = x + (xc - x).detach()  # forward=clamp, backward=identity
        # no-crop resize (왜곡 허용, 전체 보존)
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x.float(), (299, 299), mode="bilinear", align_corners=False, antialias=True)
        else:
            x = x.float()
        x = x.to(p.dtype)
        feats = self.net(x)[0].squeeze(-1).squeeze(-1)
        return feats

class InceptionClassifier(nn.Module):
    def __init__(self, num_classes=1000,
                 weights=Inception_V3_Weights.IMAGENET1K_V1,
                 freeze_backbone=True, label_smoothing=0.0,
                 net_dtype=torch.bfloat16):
        super().__init__()
        m = inception_v3(weights=weights)
        if num_classes != m.fc.out_features:
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.m = m.to(dtype=net_dtype)
        self.m.eval() if freeze_backbone else self.m.train()
        for p in self.m.parameters(): p.requires_grad_(not freeze_backbone)

        try:
            mean, std = weights.meta["mean"], weights.meta["std"]
        except Exception:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
        self.label_smoothing = float(label_smoothing)

    def forward(self, x, targets=None, input_range="-1..1", clamp="ste"):
        p = next(self.m.parameters()); x = x.to(p.device)
        if input_range == "-1..1": x = (x + 1) * 0.5
        if clamp == "hard":
            x = x.clamp(0, 1)
        elif clamp == "ste":
            x = x + (x.clamp(0, 1) - x).detach()
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x.float(), (299, 299), mode="bilinear",
                              align_corners=False, antialias=True)
        else:
            x = x.float()
        # weights 사용 시 inception 내부에서 transform_input=True → 이중 정규화 방지
        if not getattr(self.m, "transform_input", False):
            x = (x - self.mean) / self.std

        x = x.to(p.dtype)
        out = self.m(x)
        logits = out.logits if hasattr(out, "logits") else out  # aux_logits=True+train() 대비

        result = {"logits": logits, "pred": logits.argmax(1)}
        if targets is not None:
            result["loss"] = F.cross_entropy(
                logits.float(), targets.long(), label_smoothing=self.label_smoothing
            )
        return result