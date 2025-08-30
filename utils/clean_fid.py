# pip install clean-fid
import numpy as np
import torch
from torch import nn
from typing import List
from PIL import Image

# Clean-FID: 정확한 리사이즈 & Inception feature 추출기
from cleanfid.features import build_feature_extractor
from cleanfid.resize import build_resizer

class CleanFIDInception(nn.Module):
    """
    Drop-in 대체: 기존 FIDInception과 동일한 역할.
      - forward(pil_samples): List[PIL.Image] -> (N, 2048) torch.Tensor
      - encode(x): x in [-1,1], (N,3,H,W) -> (N,2048) torch.Tensor
    내부는 Clean-FID의 'clean' 모드 리사이즈/전처리를 따름.
    """
    def __init__(self, device=None, use_dataparallel: bool = False, mode: str = "clean"):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.mode = mode

        # Clean-FID가 제공하는 Inception feature 추출기(클린 모드 권장)
        # - 입력: [0,255] 범위, (N,3,H,W)
        # - 출력: (N, 2048)
        self._feat_model = build_feature_extractor(
            mode=mode, device=device, use_dataparallel=use_dataparallel
        )
        # 정확한(anti-aliased) 리사이저. 기본 출력은 299×299.
        self._resize = build_resizer(mode)

    @torch.no_grad()
    def _pil_list_to_batch(self, pil_samples: List[Image.Image]) -> torch.Tensor:
        # PIL -> RGB numpy -> Clean-FID 리사이즈 -> (N,3,299,299) torch.uint8/float32
        arrs = [np.asarray(img.convert("RGB")) for img in pil_samples]   # (H,W,3), uint8
        arrs = [self._resize(a) for a in arrs]                           # 정확 리사이즈(299×299)
        batch = np.stack([a.transpose(2, 0, 1) for a in arrs], axis=0)   # (N,3,299,299)
        batch = torch.from_numpy(batch).to(self.device)                  # [0,255]
        return batch

    @torch.no_grad()
    def forward(self, pil_samples: List[Image.Image]) -> torch.Tensor:
        batch = self._pil_list_to_batch(pil_samples)
        feats = self._feat_model(batch)  # (N,2048)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        return feats