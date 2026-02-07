import torch
import torch.nn.functional as F
from .transform import Transform

# tau_x, tau_e sigmoid 처리, tau_raw_init=-3.0
class LogLinearTransform(Transform):
    def __init__(self,
        gamma_push=False,
        gamma_max=None,
        kappa_max=None,
        tau_raw_init=-3.0,
        eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.kappa_max = kappa_max
        self.tau_raw_init = tau_raw_init
        self.eps = eps
        
    def unpack(self, params):
        # params: [B, ?] or [B, ?, C] (gamma, tau_x, tau_e가 앞 3개라고 가정)
        gamma = params[:, 0]
        tau_x = params[:, 1]
        tau_e = params[:, 2]
        k_x = params[:, 3]
        k_e = params[:, 4]

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max

        tau_x = torch.sigmoid(tau_x + self.tau_raw_init)
        tau_e = torch.sigmoid(tau_e + self.tau_raw_init)

        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)
            
        if self.kappa_max:
            k_x = torch.tanh(k_x) * self.kappa_max
            k_e = torch.tanh(k_e) * self.kappa_max

        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e, 'kappa_x': k_x, 'kappa_e': k_e}

    def L(self, log_y, y, p, side='x'):
        tau = (p['tau_x'] if side=='x' else p['tau_e'])
        return (1 - tau) * log_y + tau * y