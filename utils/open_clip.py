# pip install open_clip_torch
import torch, torch.nn.functional as F
from torch import nn
import open_clip
from typing import Tuple, Optional, List

class OpenCLIPEmbedder(nn.Module):
    """
    open_clip image/text encoder.
    - No-crop resize (bilinear), input_range '-1..1' or '0..1', 'hard'/'ste' clamp
    - bf16 OK, tokens are always Long
    - Optional L2-norm on outputs
    - Cosine / InfoNCE(CLIP) / Cos+MSE losses
    """
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        net_dtype: torch.dtype = torch.bfloat16,
        l2_normalize: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # model / preprocess / tokenizer
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model = model.eval().to(dtype=net_dtype)
        for p in self.model.parameters(): p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # infer image_size and Normalize(mean,std) used by this pretrained
        self.img_size = self._infer_image_size(self.model, preprocess)
        mean, std = self._infer_mean_std(preprocess)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1), persistent=False)

        self.l2_normalize = l2_normalize
        self.net_dtype = net_dtype

    # -------------------------- helpers --------------------------
    @staticmethod
    def _infer_image_size(model, preprocess) -> int:
        # try from model
        img_size = getattr(getattr(model, "visual", None), "image_size", None)
        if isinstance(img_size, (tuple, list)):
            img_size = int(img_size[-1])
        if isinstance(img_size, int):
            return img_size
        # fallback: scan transforms for Resize/CenterCrop size
        size = None
        tfs = getattr(preprocess, "transforms", None)
        if tfs is not None:
            for t in tfs:
                s = getattr(t, "size", None)
                if s is not None:
                    if isinstance(s, (tuple, list)): s = s[-1]
                    if isinstance(s, int):
                        size = s
        return int(size or 224)

    @staticmethod
    def _infer_mean_std(preprocess) -> Tuple[List[float], List[float]]:
        # try to find torchvision.transforms.Normalize
        tfs = getattr(preprocess, "transforms", None)
        if tfs is not None:
            for t in tfs:
                if t.__class__.__name__.lower() == "normalize":
                    mean = [float(x) for x in getattr(t, "mean", [0.48145466, 0.4578275, 0.40821073])]
                    std  = [float(x) for x in getattr(t, "std",  [0.26862954, 0.26130258, 0.27577711])]
                    return mean, std
        # fallback to OpenAI CLIP stats
        return [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]

    # -------------------------- encoders --------------------------
    def encode_image(self, x: torch.Tensor, input_range: str = "-1..1", clamp_mode: str = "ste"):
        """
        x: FloatTensor [B,C,H,W] in -1..1 or 0..1, 1ch or 3ch
        returns: [B, D]
        """
        p = next(self.model.parameters())
        x = x.to(p.device)

        # channels
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3: raise ValueError("x must have 1 or 3 channels")

        # range → 0..1
        if input_range == "-1..1": x = (x + 1) * 0.5
        elif input_range != "0..1": raise ValueError("input_range must be '-1..1' or '0..1'")

        # clamp
        if clamp_mode == "hard":
            x = x.clamp(0, 1)
        elif clamp_mode == "ste":
            x = x + (x.clamp(0, 1) - x).detach()
        else:
            raise ValueError("clamp_mode must be 'hard' or 'ste'")

        # resize (no crop)
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x.float(), (self.img_size, self.img_size),
                              mode="bilinear", align_corners=False, antialias=True)
        else:
            x = x.float()

        # normalize (per open_clip pretrained)
        mean = self.mean.to(x.device, dtype=x.dtype)
        std  = self.std.to(x.device, dtype=x.dtype)
        x = ((x - mean) / std).to(self.net_dtype)

        # model dtype/device
        x = x.to(device=self.device, dtype=p.dtype)
        feats = self.model.encode_image(x)
        if self.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def encode_text(self, texts, l2_normalize: Optional[bool] = None):
        """
        texts: str | list[str] | torch.LongTensor[B, context_len]
        returns: [B, D]
        """
        # tokens
        if isinstance(texts, torch.Tensor):
            tokens = texts.to(device=self.device, dtype=torch.long)
        else:
            if isinstance(texts, str): texts = [texts]
            tokens = self.tokenizer(texts)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.as_tensor(tokens)
            tokens = tokens.to(device=self.device, dtype=torch.long)

        feats = self.model.encode_text(tokens)
        if (self.l2_normalize if l2_normalize is None else l2_normalize):
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    # -------------------------- losses --------------------------
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
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode)
        txt = self.encode_text(texts)

        if normalize:
            img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # broadcast 1→B
        if txt.shape[0] == 1 and img.shape[0] > 1: txt = txt.expand(img.shape[0], -1)
        if img.shape[0] == 1 and txt.shape[0] > 1: img = img.expand(txt.shape[0], -1)
        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        cos = F.cosine_similarity(img.float(), txt.float(), dim=-1)  # [B]
        loss_vec = 1.0 - cos
        if reduction == "mean":   return loss_vec.mean()
        if reduction == "sum":    return loss_vec.sum()
        if reduction == "none":   return loss_vec
        raise ValueError("reduction must be 'mean'|'sum'|'none'")

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
        use_model_logit_scale: bool = False,  # True면 모델의 learnable logit_scale 사용
    ):
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode)
        txt = self.encode_text(texts)

        # L2-normalize (standard CLIP)
        img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        if use_model_logit_scale and hasattr(self.model, "logit_scale"):
            logit_scale = self.model.logit_scale.exp().detach().item()
        else:
            logit_scale = 1.0 / float(temperature)

        logits_per_image = (logit_scale * img @ txt.t()).float()   # [B,B]
        logits_per_text  = logits_per_image.t().contiguous()       # [B,B]
        targets = torch.arange(img.shape[0], device=logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, targets, reduction=reduction)
        if symmetric:
            loss_t = F.cross_entropy(logits_per_text,  targets, reduction=reduction)
            loss = 0.5 * (loss_i + loss_t)
        else:
            loss = loss_i

        if return_logits:
            return loss, {"logits_per_image": logits_per_image, "logits_per_text": logits_per_text}
        return loss

    def get_cossim_plus_mse_loss(
        self,
        images,
        texts,
        *,
        input_range: str = "-1..1",
        clamp_mode: str = "ste",
        normalize_for_cos: bool = True,
        mse_after_norm: bool = False,   # False: unnormalized MSE(추천), True: normalized MSE(=cos와 중복)
        w_cos: float = 1.0,
        w_mse: float = 0.1,
        reduction: str = "mean",
        return_parts: bool = False,
    ):
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode)
        txt = self.encode_text(texts)

        # broadcast 1→B
        if txt.shape[0] == 1 and img.shape[0] > 1: txt = txt.expand(img.shape[0], -1)
        if img.shape[0] == 1 and txt.shape[0] > 1: img = img.expand(txt.shape[0], -1)
        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        if normalize_for_cos:
            img_n = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt_n = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            img_n, txt_n = img, txt

        cos = F.cosine_similarity(img_n.float(), txt_n.float(), dim=-1)  # [B]
        cos_loss = 1.0 - cos

        a, b = (img_n, txt_n) if mse_after_norm else (img, txt)
        mse_loss = F.mse_loss(a.float(), b.float(), reduction="none").mean(dim=-1)  # [B]

        total = w_cos * cos_loss + w_mse * mse_loss
        if reduction == "mean": total = total.mean()
        elif reduction == "sum": total = total.sum()
        elif reduction != "none": raise ValueError("reduction must be 'mean'|'sum'|'none'")

        if return_parts:
            return total, {"cos": cos_loss, "mse": mse_loss}
        return total
