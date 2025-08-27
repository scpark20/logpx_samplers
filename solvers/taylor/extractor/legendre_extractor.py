import torch
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, steps=5, out_dim=5, n_channels=1, **kwargs):
        super().__init__()
        self.steps = steps
        self.n_channels = n_channels
        # coeff[0]: P0, coeff[1]: P1, coeff[2]: P2
        self.coeff = nn.Parameter(torch.zeros(3, 2, out_dim))

    def forward(self, inputs):
        # step -> s in [0,1]  (off-by-one 방지: steps-1 사용)
        step = inputs['step']
        if torch.is_tensor(step):
            s = step.to(self.coeff.dtype, self.coeff.device) / max(1, self.steps - 1)
        else:
            s = torch.tensor(step / max(1, self.steps - 1),
                             dtype=self.coeff.dtype, device=self.coeff.device)
        s = s.view(1, 1, 1)

        # Legendre basis on [0,1]
        P0 = s.new_tensor(1.0)
        P1 = 2*s - 1
        P2 = 6*s*s - 6*s + 1

        c = self.coeff  # (3, 2, out_dim)
        out = c[0:1]*P0 + c[1:2]*P1 + c[2:3]*P2  # (1, 2, out_dim)

        return out, None
