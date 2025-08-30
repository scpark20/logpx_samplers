import torch
from torch import nn
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as TF

class FIDInception(nn.Module):
    def __init__(self, dims=2048, resize_input=True, normalize_input=True, device=None, net_dtype=torch.float32):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]],
                                normalize_input=normalize_input
                                ).eval().to(device=device, dtype=net_dtype)
        for p in self.net.parameters(): p.requires_grad_(False)
        self.transform = TF.ToTensor()

    def forward(self, pil_samples):
        samples = torch.stack([self.transform(sample) for sample in pil_samples])
        p = next(self.net.parameters())
        samples = samples.to(p.device, p.dtype)
        feats = self.net(samples)[0][:, :, 0, 0]
        return feats

    def encode(self, x):
        x = x.clamp(-1, 1)
        p = next(self.net.parameters())
        x = x.to(p.device, p.dtype)
        with torch.no_grad():
            return self.net(x)[0][:, :, 0, 0]

            