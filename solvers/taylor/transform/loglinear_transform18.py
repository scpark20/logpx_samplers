import torch
import torch.nn.functional as F
from .transform import Transform

def ste_clamp_01(x, eps=1e-8):
    y = torch.clamp(x, 0.0 + eps, 1.0 - eps)
    #return y.detach() + x - x.detach()
    return x + (y - x).detach()

# ste clamp + tanh * 2
class LogLinearTransform(Transform):
    def __init__(self,
        gamma_push=False,
        eps=1e-2):
        self.gamma_push = gamma_push
        self.eps = eps
        
    def unpack(self, params):
        # params: [B, ?] or [B, ?, C] (gamma, tau_x, tau_e가 앞 3개라고 가정)
        gamma = params[:, 0]
        tau_x = params[:, 1]
        tau_e = params[:, 2]
        k_x = params[:, 3]
        k_e = params[:, 4]

        # print('self.gamma_init :', self.gamma_init)
        # print('self.tau_init :', self.tau_init)

        gamma = torch.tanh(gamma) * 2
        tau_x = ste_clamp_01(tau_x + 1.0)
        tau_e = ste_clamp_01(tau_e + 1.0)
        
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        k_x = torch.tanh(k_x) * 2
        k_e = torch.tanh(k_e) * 2
        
        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e, 'kappa_x': k_x, 'kappa_e': k_e}

    def L(self, log_y, y, p, side='x'):
        tau = (p['tau_x'] if side=='x' else p['tau_e'])
        return (1 - tau) * log_y + tau * y