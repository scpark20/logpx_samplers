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
        temperature: float | None = 0.07,   # ← 기본값 0.07 유지
        input_range: str = "-1..1",
        clamp_mode: str = "ste",
        symmetric: bool = True,
        reduction: str = "mean",
        return_logits: bool = False,
    ):
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode)
        txt = self.encode_text(texts)
        img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        # --- scale = 1/tau 결정 ---
        if temperature is None:
            if hasattr(self.model, "logit_scale"):
                ls = self.model.logit_scale
                val = float(ls.detach().to(torch.float32))
                scale = ls.exp() if val < 10.0 else ls   # log-파라미터/선형스케일 모두 대응
                scale = torch.clamp(scale, max=100)      # 관례(τ ≥ 0.01)
            else:
                raise ValueError("Model has no 'logit_scale'; pass an explicit temperature.")
        else:
            scale = 1.0 / float(temperature)

        if not torch.is_tensor(scale):
            scale = torch.as_tensor(scale, device=img.device, dtype=img.dtype)
        else:
            scale = scale.to(device=img.device, dtype=img.dtype)

        logits_per_image = (scale * img @ txt.t()).float()
        logits_per_text  = logits_per_image.t().contiguous()
        targets = torch.arange(img.shape[0], device=logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, targets, reduction=reduction)
        loss = 0.5 * (loss_i + F.cross_entropy(logits_per_text, targets, reduction=reduction)) if symmetric else loss_i
        return (loss, {"logits_per_image": logits_per_image, "logits_per_text": logits_per_text}) if return_logits else loss

    def cosine_infonce_like(self, images, texts, *,
                        tau=0.07,
                        neg_mean=None,   # None이면 배치로 추정(없으면 0.0)
                        eff_batch=None,  # None이면 실제 B 사용
                        input_range="-1..1", clamp_mode="ste",
                        reduction="mean", loss_weight=1.0, eps=1e-6):
        img, txt, s_pos = self._pair_norm_(images, texts, input_range, clamp_mode, eps)  # s_pos: [B]
        B = s_pos.shape[0]
        if eff_batch is None: eff_batch = max(B, 2)  # log(B-1) 안전
        if neg_mean is None:
            if B > 1:
                sims = (img @ txt.t()).float()               # [B,B]
                neg_mean = sims[~torch.eye(B, dtype=torch.bool, device=sims.device)].mean()
            else:
                neg_mean = s_pos.new_tensor(0.0)

        m_eff = neg_mean + tau * torch.log(s_pos.new_tensor(max(eff_batch-1, 1.0)))
        z = (m_eff - s_pos) / max(tau, 1e-6)                # temperature 적용
        loss_vec = F.softplus(z)                             # logistic형
        if reduction == "mean": loss = loss_vec.mean()
        elif reduction == "sum": loss = loss_vec.sum()
        else: loss = loss_vec
        return loss_weight * loss

    # ---------------------------------------------------------------------
    # 3) Angular / Hinge / Sharpened Cosine losses
    #    - 모든 손실은 내부에서 L2 정규화 후 fp32로 계산하여 안정화합니다.
    #    - texts 가 1개이고 images가 B개면 자동 broadcast 합니다.
    #    - 각도 단위는 라디안입니다.
    # ---------------------------------------------------------------------

    def angular_loss(
        self,
        images,
        texts,
        *,
        input_range: str = "-1..1",
        clamp_mode: str = "ste",
        squared: bool = False,        # True면 θ^2 (근접 구간 더 부드러움)
        reduction: str = "mean",
        loss_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        L_ang = arccos( cos(u,v) )  (정규화된 임베딩)
        ||∂L/∂u|| = 1 이라 근접할수록 사라지지 않는 안정적 신호.
        """
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode).float()
        txt = self.encode_text(texts).float()

        # L2 normalize
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)

        # broadcast 1→B if needed
        if txt.shape[0] == 1 and img.shape[0] > 1:
            txt = txt.expand(img.shape[0], -1)
        if img.shape[0] == 1 and txt.shape[0] > 1:
            img = img.expand(txt.shape[0], -1)
        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        s = (img * txt).sum(dim=-1).clamp_(-1 + eps, 1 - eps)
        theta = torch.acos(s)                           # [0, π]
        loss_vec = theta * theta if squared else theta

        if reduction == "mean": loss = loss_vec.mean()
        elif reduction == "sum": loss = loss_vec.sum()
        elif reduction == "none": loss = loss_vec
        else: raise ValueError("reduction must be 'mean'|'sum'|'none'")
        return loss_weight * loss

    def angular_hinge(
        self,
        images,
        texts,
        *,
        input_range: str = "-1..1",
        clamp_mode: str = "ste",
        margin: float = 0.10,         # 라디안: ~5.7°
        reduction: str = "mean",
        loss_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        L = relu( arccos(cos) - margin )
        -> θ가 margin 이하로 내려갈 때까지 지속적으로 신호를 줌.
        """
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode).float()
        txt = self.encode_text(texts).float()

        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)

        if txt.shape[0] == 1 and img.shape[0] > 1:
            txt = txt.expand(img.shape[0], -1)
        if img.shape[0] == 1 and txt.shape[0] > 1:
            img = img.expand(txt.shape[0], -1)
        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        s = (img * txt).sum(dim=-1).clamp_(-1 + eps, 1 - eps)
        theta = torch.acos(s)
        loss_vec = F.relu(theta - margin)

        if reduction == "mean": loss = loss_vec.mean()
        elif reduction == "sum": loss = loss_vec.sum()
        elif reduction == "none": loss = loss_vec
        else: raise ValueError("reduction must be 'mean'|'sum'|'none'")
        return loss_weight * loss

    def cosine_sharp_loss(
        self,
        images,
        texts,
        *,
        input_range: str = "-1..1",
        clamp_mode: str = "ste",
        tau: float = 0.85,            # 상단 영역 확장 임계(0.8~0.9 권장)
        alpha: float = 8.0,           # softplus 기울기(5~10 권장)
        reduction: str = "mean",
        loss_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        코사인 근접 구간을 '확대'하여 좋은 품질에서도 그래디언트 유지:
          s_t = (s - tau) / (1 - tau)
          L   = softplus(alpha * (1 - s_t)) / alpha
        """
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode).float()
        txt = self.encode_text(texts).float()

        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)

        if txt.shape[0] == 1 and img.shape[0] > 1:
            txt = txt.expand(img.shape[0], -1)
        if img.shape[0] == 1 and txt.shape[0] > 1:
            img = img.expand(txt.shape[0], -1)
        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"Batch mismatch: images={img.shape[0]} vs texts={txt.shape[0]}")

        s = (img * txt).sum(dim=-1).clamp_(-1 + eps, 1 - eps)
        s_t = (s - tau) / (1 - tau + 1e-6)             # 상단 영역 재스케일
        loss_vec = F.softplus(alpha * (1.0 - s_t)) / alpha

        if reduction == "mean": loss = loss_vec.mean()
        elif reduction == "sum": loss = loss_vec.sum()
        elif reduction == "none": loss = loss_vec
        else: raise ValueError("reduction must be 'mean'|'sum'|'none'")
        return loss_weight * loss

    def _pair_norm_(self, images, texts, input_range="-1..1", clamp_mode="ste", eps=1e-6):
        img = self.encode_image(images, input_range=input_range, clamp_mode=clamp_mode).float()
        txt = self.encode_text(texts).float()
        img = F.normalize(img, dim=-1); txt = F.normalize(txt, dim=-1)
        if txt.shape[0]==1 and img.shape[0]>1: txt = txt.expand(img.shape[0], -1)
        if img.shape[0]==1 and txt.shape[0]>1: img = img.expand(txt.shape[0], -1)
        if img.shape[0]!=txt.shape[0]: raise ValueError(f"B mismatch: img={img.shape[0]} txt={txt.shape[0]}")
        s = (img*txt).sum(dim=-1).clamp(-1+eps, 1-eps)
        return img, txt, s

    # 4) Sine loss
    def sine_loss(self, images, texts, *, input_range="-1..1", clamp_mode="ste",
                reduction="mean", loss_weight=1.0, eps=1e-6):
        _, _, s = self._pair_norm_(images, texts, input_range, clamp_mode, eps)
        loss_vec = torch.sqrt(1.0 - s*s)  # sin(theta)
        return loss_weight * (loss_vec.mean() if reduction=="mean" else loss_vec.sum() if reduction=="sum" else loss_vec)
