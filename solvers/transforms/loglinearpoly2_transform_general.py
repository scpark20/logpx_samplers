import torch
from .transform import Transform

class LogLinearPoly2Transform(Transform):
    def __init__(self, gamma_max=None, kappa_max=None, poly_max=None, gamma_push=False, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.kappa_max = kappa_max
        self.poly_max = poly_max
        self.eps = eps

    def unpack(self, params):
        # params: [B, 5]
        gamma   = params[:, 0]
        tau_x   = params[:, 1]
        tau_e   = params[:, 2]
        p_x   = params[:, 3]
        p_e   = params[:, 4]
        q_x   = params[:, 5]
        q_e   = params[:, 6]
        kappa_x = params[:, 7]
        kappa_e = params[:, 8]

        tau_x = torch.sigmoid(tau_x)
        tau_e = torch.sigmoid(tau_e)
        
        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max

        if self.kappa_max is not None:
            kappa_x = torch.tanh(kappa_x) * self.kappa_max
            kappa_e = torch.tanh(kappa_e) * self.kappa_max

        if self.poly_max is not None:
            p_x = torch.sigmoid(p_x) * self.poly_max
            p_e = torch.sigmoid(p_e) * self.poly_max
            q_x = torch.sigmoid(q_x) * self.poly_max
            q_e = torch.sigmoid(q_e) * self.poly_max
        else:
            p_x = torch.softplus(p_x)
            p_e = torch.softplus(p_e)
            q_x = torch.softplus(q_x)
            q_e = torch.softplus(q_e)
            
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
                'p_x': p_x, 'p_e': p_e, 'q_x': q_x, 'q_e': q_e,
                'kappa_x': kappa_x, 'kappa_e': kappa_e}

    def L(self, log_y, y, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p = params['p_x'] if side=='x' else params['p_e']
        q = params['q_x'] if side=='x' else params['q_e']
        return (1.0 - tau) * q * log_y + tau * (y ** p)
    
    def g(self, y, params, side='x'):
        tau = params['tau_x'] if side == 'x' else params['tau_e']
        p   = params['p_x']   if side == 'x' else params['p_e']
        q   = params['q_x']   if side == 'x' else params['q_e']
        return y / ((1.0 - tau) * q + tau * p * (y ** p))