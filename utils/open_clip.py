# pip install open_clip_torch
import torch, torch.nn.functional as F
from torch import nn
import open_clip

class OpenCLIPEmbedder(nn.Module):
    """open_clip image/text encoder. No-crop resize, -1..1→[0,1], 'hard'/'ste' clamp, net_dtype cast, optional L2-norm."""
    def __init__(self, model_name="ViT-B-16", pretrained="openai",
                 net_dtype=torch.bfloat16, l2_normalize=False, device=None, tokenizer_name=None):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.model.eval().to(device=self.device, dtype=net_dtype)   # ← 가중치 dtype을 net_dtype로
        for p in self.model.parameters(): p.requires_grad_(False)

        size = getattr(getattr(self.model, "visual", object()), "image_size", 224)
        self.img_size = int(size[0] if isinstance(size, (tuple, list)) else size)

        mean = getattr(getattr(self.model, "visual", object()), "image_mean",
                       [0.48145466, 0.4578275, 0.40821073])
        std  = getattr(getattr(self.model, "visual", object()), "image_std",
                       [0.26862954, 0.26130258, 0.27577711])
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(std ).view(1,3,1,1), persistent=False)

        self.tokenizer = open_clip.get_tokenizer(tokenizer_name or model_name)
        self.l2_normalize, self.net_dtype = l2_normalize, net_dtype

    def encode_image(self, x, input_range="-1..1", clamp_mode="ste"):
        if x.shape[1] == 1: x = x.repeat(1,3,1,1)
        elif x.shape[1] != 3: raise ValueError("x must have 1 or 3 channels")

        if input_range == "-1..1": x = (x + 1) * 0.5
        elif input_range != "0..1": raise ValueError("input_range must be '-1..1' or '0..1'")

        if clamp_mode == "hard": x = x.clamp(0,1)
        elif clamp_mode == "ste": x = x + (x.clamp(0,1) - x).detach()
        else: raise ValueError("clamp_mode must be 'hard' or 'ste'")

        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x.float(), (self.img_size, self.img_size),
                              mode="bilinear", align_corners=False, antialias=True)
        else:
            x = x.float()

        x = ((x - self.mean) / self.std).to(device=self.device, dtype=self.net_dtype)
        feats = self.model.encode_image(x)
        if self.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def encode_text(self, texts, l2_normalize=None):
        """texts: str | list[str] | torch.LongTensor[B,ctx] → (B,D)"""
        if isinstance(texts, torch.Tensor):
            tokens = texts.to(device=self.device, dtype=torch.long)   # 토큰은 Long 유지
        else:
            if isinstance(texts, str): texts = [texts]
            tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        if (self.l2_normalize if l2_normalize is None else l2_normalize):
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats
