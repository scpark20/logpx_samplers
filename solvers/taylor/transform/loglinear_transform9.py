import torch
import torch.nn.functional as F
from .transform import Transform

def ste_clamp_01(x):
    y = torch.clamp(x, 0.0, 1.0)
    return y.detach() + x - x.detach()

# tau_x, tau_e init값 주고 ste_clamp
class LogLinearTransform(Transform):
    def __init__(self,
        gamma_push=False,
        gamma_init=0.0,
        tau_init=0.0,
        eps=1e-2):
        self.gamma_push = gamma_push
        self.eps = eps
        self.gamma_init = gamma_init
        self.tau_init = tau_init
        
    def unpack(self, params):
        # params: [B, ?] or [B, ?, C] (gamma, tau_x, tau_e가 앞 3개라고 가정)
        gamma = params[:, 0]
        tau_x = params[:, 1]
        tau_e = params[:, 2]
        k_x = params[:, 3]
        k_e = params[:, 4]

        # print('self.gamma_init :', self.gamma_init)
        # print('self.tau_init :', self.tau_init)

        gamma = gamma + self.gamma_init
        tau_x = ste_clamp_01(tau_x + self.tau_init)
        tau_e = ste_clamp_01(tau_e + self.tau_init)
        
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)
        
        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e, 'kappa_x': k_x, 'kappa_e': k_e}

    def L(self, log_y, y, p, side='x'):
        tau = (p['tau_x'] if side=='x' else p['tau_e'])
        return (1 - tau) * log_y + tau * y