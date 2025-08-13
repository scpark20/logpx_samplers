import torch, torch.nn.functional as F
from torch import nn
from pytorch_fid.inception import InceptionV3

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