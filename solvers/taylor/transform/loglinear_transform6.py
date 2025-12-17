import torch
import torch.nn.functional as F
from .transform import Transform

# tau_x, tau_e init값 각각 주고 sigmoid
class LogLinearTransform(Transform):
    def __init__(self,
        gamma_push=False,
        gamma_init=0.0,
        tau_x_init=0.0,
        tau_e_init=0.0,
        eps=1e-2):
        self.gamma_push = gamma_push
        self.eps = eps
        self.gamma_init = gamma_init
        self.tau_x_init = tau_x_init
        self.tau_e_init = tau_e_init
        
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
        tau_x = torch.sigmoid(tau_x + self.tau_x_init)
        tau_e = torch.sigmoid(tau_e + self.tau_e_init)
        
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)
        
        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e, 'kappa_x': k_x, 'kappa_e': k_e}

    def L(self, log_y, y, p, side='x'):
        tau = (p['tau_x'] if side=='x' else p['tau_e'])
        return (1 - tau) * log_y + tau * y