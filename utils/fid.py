import torch
from torch import nn
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as TF

class FIDInception(nn.Module):
    def __init__(self, dims=2048, net_dtype=torch.float32):
        super().__init__()
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).eval().to(dtype=net_dtype)
        for p in self.net.parameters(): p.requires_grad_(False)
        self.transform = TF.ToTensor()

    def forward(self, pil_samples):
        samples = torch.stack([self.transform(sample) for sample in pil_samples])
        p = next(self.net.parameters())
        samples = samples.to(p.device, p.dtype)
        feats = self.net(samples)[0][:, :, 0, 0]
        return feats
