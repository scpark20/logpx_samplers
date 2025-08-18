# pip install "git+https://github.com/openai/CLIP.git"
import torch, torch.nn.functional as F
from torch import nn
import clip

class CLIPEmbedder(nn.Module):
    """openai/CLIP image/text encoder. No-crop resize, -1..1→[0,1], 'hard'/'ste' clamp, bf16 OK, optional L2-norm."""
    def __init__(self, model_name="ViT-B/16", net_dtype=torch.bfloat16, l2_normalize=False, device=None):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _ = clip.load(model_name, device=self.device, jit=False)
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
        x = x.to(device=self.device, dtype=p.dtype)
        feats = self.model.encode_image(x)
        if self.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    # --- FIXED: keep tokens as long (no dtype cast to bf16) ---
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

        # NOTE: tokens must be long for CLIP
        tokens = tokens.to(device=self.device, dtype=torch.long)

        feats = self.model.encode_text(tokens)
        if (self.l2_normalize if l2_normalize is None else l2_normalize):
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    # ---------------------------------------------------------------------
    # 1) Cosine-similarity loss for matched (image, text) pairs
    #    loss = mean(1 - cos(img_i, txt_i))
    # ---------------------------------------------------------------------
    def get_cossim_loss(
        self,
        images,
        texts,
        *,
        input_range: str = "-1..1",
        clamp_mode: str = "ste",
        normalize: bool = True,
        reduction: str = "mean",
    ):
        """
        images:  FloatTensor [B, C, H, W]  in -1..1 or 0..1
        texts:   str | list[str] | LongTensor[B,77]
        returns: scalar loss (or [B] if reduction='none')
        """
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode)
        txt = self.encode_text(texts)

        if normalize:
            img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # broadcast 1→B if 필요
        if txt.shape[0] == 1 and img.shape[0] > 1:
            txt = txt.expand(img.shape[0], -1)
        if img.shape[0] == 1 and txt.shape[0] > 1:
            img = img.expand(txt.shape[0], -1)
        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        cos = F.cosine_similarity(img.float(), txt.float(), dim=-1)  # [B]
        loss_vec = 1.0 - cos
        if reduction == "mean":   return loss_vec.mean()
        if reduction == "sum":    return loss_vec.sum()
        if reduction == "none":   return loss_vec
        raise ValueError("reduction must be 'mean'|'sum'|'none'")

    # ---------------------------------------------------------------------
    # 2) CLIP contrastive loss (InfoNCE) as in the paper
    #    logits = (img @ txt^T) / tau, tau=temperature
    #    loss = (CE(img->text) + CE(text->img)) / 2  if symmetric=True
    # ---------------------------------------------------------------------
    def get_clip_loss(
        self,
        images,
        texts,
        *,
        temperature: float = 0.07,
        input_range: str = "-1..1",
        clamp_mode: str = "ste",
        symmetric: bool = True,
        reduction: str = "mean",
        return_logits: bool = False,
    ):
        """
        images:  FloatTensor [B, C, H, W]
        texts:   str | list[str] | LongTensor[B,77]
        temperature: tau in (0, +inf). Smaller → sharper.
        returns: loss (and optionally logits dict)
        """
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode)
        txt = self.encode_text(texts)

        # L2 normalize for cosine-sim based logits (standard CLIP training)
        img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        logit_scale = 1.0 / float(temperature)
        logits_per_image = (logit_scale * img @ txt.t()).float()         # [B,B]
        logits_per_text  = logits_per_image.t().contiguous()             # [B,B]
        targets = torch.arange(img.shape[0], device=logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, targets, reduction=reduction)
        if symmetric:
            loss_t = F.cross_entropy(logits_per_text,  targets, reduction=reduction)
            if reduction == "none":
                loss = 0.5 * (loss_i + loss_t)                            # [B]
            else:
                loss = 0.5 * (loss_i + loss_t)                            # scalar
        else:
            loss = loss_i

        if return_logits:
            return loss, {"logits_per_image": logits_per_image, "logits_per_text": logits_per_text}
        return loss
