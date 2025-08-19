# pip install "transformers>=4.40" timm
import torch, torch.nn.functional as F
from torch import nn
from transformers import BlipForConditionalGeneration, AutoTokenizer

class BLIPTextLikelihood(nn.Module):
    """
    - 이미지→텍스트 우도 기반 손실: NLL(y|I) 및 PMI(y;I) = NLL(y|null) - NLL(y|I)
    - 전처리/리사이즈/정규화를 torch로 처리(grad 보존), 토큰은 Long 유지
    """
    def __init__(self, model_name="Salesforce/blip-image-captioning-base",
                 net_dtype=torch.bfloat16, device=None, max_length=64):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model.eval().to(dtype=net_dtype)
        for p in self.model.parameters(): p.requires_grad_(False)

        self.img_size = int(self.model.config.vision_config.image_size)  # 보통 384
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        self.net_dtype, self.maxlen = net_dtype, max_length

    # ---- 이미지 전처리(grad 유지): [-1,1]/[0,1] -> clamp -> resize -> norm ----
    def _preprocess_image(self, x, input_range="-1..1", clamp_mode="ste"):
        x = x.to(self.device)
        if x.shape[1] == 1: x = x.repeat(1,3,1,1)
        elif x.shape[1] != 3: raise ValueError("x must have 1 or 3 channels")
        if input_range == "-1..1": x = (x + 1) * 0.5
        elif input_range != "0..1": raise ValueError("input_range must be '-1..1' or '0..1'")
        x = x + (x.clamp(0,1) - x).detach() if clamp_mode=="ste" else x.clamp(0,1)
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x.float(), (self.img_size, self.img_size),
                              mode="bilinear", align_corners=False, antialias=True)
        else:
            x = x.float()
        return ((x - self.mean) / self.std).to(dtype=self.net_dtype)  # pixel_values

    # ---- NLL(y|I): 텍스트 우도 기반 손실 (length_normalize 옵션 포함) ----
    def _forward_logits(self, pixel_values, input_ids):
        # labels로만 넣으면 loss는 나오지만 reduction 제어를 위해 logits도 얻어둔다
        return self.model(pixel_values=pixel_values, input_ids=input_ids, return_dict=True).logits

    def text_nll(self, images, texts, *, input_range="-1..1", clamp_mode="ste",
                 reduction="mean", length_normalize=True):
        """
        images: FloatTensor [B,C,H,W]
        texts : str | list[str] | LongTensor[B,L]
        return: NLL(y|I) (스칼라/배치/시퀀스별)
        """
        pixel = self._preprocess_image(images, input_range, clamp_mode)
        if isinstance(texts, torch.Tensor):
            ids = texts.to(self.device, dtype=torch.long)
        else:
            if isinstance(texts, str): texts = [texts]
            pack = self.tok(texts, padding=True, truncation=True,
                            max_length=self.maxlen, return_tensors="pt")
            ids = pack["input_ids"].to(self.device)

        # CE 계산을 위한 labels (-100: ignore)
        labels = ids.clone()
        if self.tok.pad_token_id is not None:
            labels[labels == self.tok.pad_token_id] = -100

        # logits: [B, L, V], teacher forcing(shift는 CE에서 처리)
        logits = self._forward_logits(pixel, ids)[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()  # 다음 토큰 예측

        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=-100, reduction="none"
        ).view(target.size())  # [B, L-1]

        mask = (target != -100).float()  # 유효 토큰
        if length_normalize:
            per_seq = (per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # 평균 NLL
        else:
            per_seq = (per_tok * mask).sum(dim=1)  # 합 NLL(길이 편향 있음)

        if reduction == "mean": return per_seq.mean()
        if reduction == "sum":  return per_seq.sum()
        if reduction == "none": return per_seq
        raise ValueError("reduction must be 'mean'|'sum'|'none'")

    # ---- PMI 기반(언어모델 prior 보정):  L = NLL(y|null) - NLL(y|I) ----
    def pmi_loss(self, images, texts, *, null_strategy="zeros", **kwargs):
        loss_pos = self.text_nll(images, texts, **kwargs, reduction="mean")
        if null_strategy == "zeros":
            null_img = torch.zeros_like(images)
        elif null_strategy == "mean":
            null_img = images.mean(dim=(2,3), keepdim=True).expand_as(images)
        else:
            raise ValueError("null_strategy ∈ {'zeros','mean'}")
        loss_null = self.text_nll(null_img, texts, **kwargs, reduction="mean")
        # PMI는 -log p(y|I) + log p(y|null) → 값이 작을수록(더 음수) 정합 ↑
        return loss_pos - loss_null
