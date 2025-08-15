# pip install "git+https://github.com/openai/CLIP.git"
import torch, torch.nn.functional as F
from torch import nn
import clip

class CLIPEmbedder(nn.Module):
    """openai/CLIP image encoder. No-crop resize, -1..1→[0,1], 'hard'/'ste' clamp, bf16 OK, optional L2-norm."""
    def __init__(self, model_name="ViT-B/16", net_dtype=torch.bfloat16, l2_normalize=False, device=None):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device or "cpu", jit=False)
        self.model.eval().to(dtype=net_dtype)
        for p in self.model.parameters(): p.requires_grad_(False)
        self.img_size = int(getattr(self.model.visual, "input_resolution", 224))
        self.l2_normalize, self.net_dtype = l2_normalize, net_dtype

    def encode_image(self, x, input_range="-1..1", clamp_mode="ste"):
        p = next(self.model.parameters()); x = x.to(p.device)
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

        mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        std  = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
        x = ((x - mean) / std).to(self.net_dtype)

        feats = self.model.encode_image(x)
        if self.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    # NEW: text → embedding
    def encode_text(self, texts, l2_normalize=None):
        """
        texts: str | list[str] | torch.LongTensor[B,77] (이미 tokenized)
        returns: (B, D) text embedding
        """
        p = next(self.model.parameters())
        device = p.device

        # 토큰 준비
        if isinstance(texts, torch.Tensor):
            # 이미 토크나이즈된 경우 (expect shape [B,77], dtype long)
            tokens = texts.to(device=device, dtype=torch.long)
        else:
            if isinstance(texts, str): texts = [texts]
            tokens = clip.tokenize(texts, truncate=True).to(device)

        feats = self.model.encode_text(tokens)
        if (self.l2_normalize if l2_normalize is None else l2_normalize):
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats