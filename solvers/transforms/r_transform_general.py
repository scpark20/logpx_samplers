import torch
from .transform import Transform

class RTransform(Transform):
    def __init__(self,
    gamma_max=None,
    tau_offset=0,
    gamma_push=False,
    eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.tau_offset = tau_offset
        self.eps = eps
        
    def unpack(self, params):
        # params: [B, 5]
        gamma   = params[:, 0]
        tau_x   = params[:, 1]
        tau_e   = params[:, 2]
        r_x = params[:, 3]
        r_e = params[:, 4]

        tau_x = torch.sigmoid(tau_x + self.tau_offset)
        tau_e = torch.sigmoid(tau_e + self.tau_offset)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        r_x = torch.exp(r_x)
        r_e = torch.exp(r_e)

        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
                'r_x': r_x, 'r_e': r_e}
    