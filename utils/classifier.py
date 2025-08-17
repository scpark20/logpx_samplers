# pip install timm  (한 번만)
import torch, torch.nn.functional as F
from torch import nn

class ViTClassifier(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
                 pretrained=True, num_classes=1000,
                 freeze_backbone=True, label_smoothing=0.0,
                 net_dtype=torch.bfloat16):
        super().__init__()
        import timm
        self.m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.m.to(dtype=net_dtype)
        self.m.eval() if freeze_backbone else self.m.train()
        for p in self.m.parameters(): p.requires_grad_(not freeze_backbone)

        # timm가 제공하는 입력 설정(해당 모델에 맞는 mean/std/해상도)
        from timm.data import resolve_model_data_config
        cfg = resolve_model_data_config(self.m)
        mean = cfg.get("mean", (0.485,0.456,0.406))
        std  = cfg.get("std",  (0.229,0.224,0.225))
        size = cfg.get("input_size", (3,224,224))[1:]
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
        self.size = (int(size[0]), int(size[1]))
        self.label_smoothing = float(label_smoothing)

    def forward(self, x, targets=None, input_range="-1..1", clamp="ste"):
        p = next(self.m.parameters()); x = x.to(p.device)
        if input_range == "-1..1": x = (x + 1) * 0.5
        if clamp == "hard":
            x = x.clamp(0, 1)
        elif clamp == "ste":
            x = x + (x.clamp(0, 1) - x).detach()
        if x.shape[-2:] != self.size:
            x = F.interpolate(x.float(), self.size, mode="bilinear",
                              align_corners=False, antialias=True)
        else:
            x = x.float()
        x = ((x - self.mean) / self.std).to(p.dtype)

        logits = self.m(x)                         # [B, num_classes]
        out = {"logits": logits, "pred": logits.argmax(1)}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits.float(), targets.long(),
                                          label_smoothing=self.label_smoothing)
        return out
