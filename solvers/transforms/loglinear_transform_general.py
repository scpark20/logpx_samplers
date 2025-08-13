import torch
from .transform import Transform

class LogLinearTransform(Transform):
    def __init__(self, gamma_max=None, kappa_max=None, gamma_push=False, kappa_disable=False, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.kappa_max = kappa_max
        self.eps = eps
        self.kappa_disable = kappa_disable

    def unpack(self, params):
        # params: [B, 5]
        gamma   = params[:, 0]
        tau_x   = params[:, 1]
        tau_e   = params[:, 2]
        kappa_x = params[:, 3]
        kappa_e = params[:, 4]

        if self.kappa_disable:
            kappa_x = kappa_x.detach()
            kappa_e = kappa_e.detach()

        tau_x = torch.sigmoid(tau_x)
        tau_e = torch.sigmoid(tau_e)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.kappa_max is not None:
            kappa_x = torch.tanh(kappa_x) * self.kappa_max
            kappa_e = torch.tanh(kappa_e) * self.kappa_max

        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
                'kappa_x': kappa_x, 'kappa_e': kappa_e}
    
    def L(self, log_y, y, p, side='x'):
        tau = p['tau_x'] if side=='x' else p['tau_e']
        return (1.0 - tau) * y + tau * log_y
    
    def g(self, y, p, side='x'):
        tau = p['tau_x'] if side=='x' else p['tau_e']
        return y / ((1.0 - tau) * y + tau)