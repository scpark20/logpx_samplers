import torch
import torch.nn.functional as F
from .transform import Transform

class TwoPowerMixTransform(Transform):
    """
    L = (1-τ) a y^{p1} + τ b y^{p2}
    g = y / ((1-τ) a p1 y^{p1} + τ b p2 y^{p2})
    """
    def __init__(self, gamma_max=3, kappa_max=3,
                 pow_max=3, amp_max=3, gamma_push=True, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max  = gamma_max
        self.kappa_max  = kappa_max
        self.pow_max    = pow_max   # 상한(선택) for p1, p2
        self.amp_max    = amp_max   # 상한(선택) for a, b
        self.eps        = eps

    def unpack(self, params):
        # params: [B, 13] = [gamma, tau_x, tau_e,
        #                     p1_x, p1_e, p2_x, p2_e,
        #                     a_x, a_e, b_x, b_e,
        #                     kappa_x, kappa_e]
        (gamma, tau_x, tau_e,
         p1_x, p1_e, p2_x, p2_e,
         a_x, a_e, b_x, b_e,
         kappa_x, kappa_e) = [params[:, i] for i in range(13)]

        tau_x = torch.sigmoid(tau_x); tau_e = torch.sigmoid(tau_e)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.kappa_max is not None:
            kappa_x = torch.tanh(kappa_x) * self.kappa_max
            kappa_e = torch.tanh(kappa_e) * self.kappa_max

        eps = torch.as_tensor(self.eps, dtype=gamma.dtype, device=gamma.device)

        # p1, p2 양수 보장 (상한 없으면 softplus, 상한 있으면 sigmoid*max)
        if self.pow_max is None:
            p1_x = F.softplus(p1_x) + eps;  p1_e = F.softplus(p1_e) + eps
            p2_x = F.softplus(p2_x) + eps;  p2_e = F.softplus(p2_e) + eps
        else:
            sig = torch.sigmoid
            p1_x = sig(p1_x) * self.pow_max + eps;  p1_e = sig(p1_e) * self.pow_max + eps
            p2_x = sig(p2_x) * self.pow_max + eps;  p2_e = sig(p2_e) * self.pow_max + eps

        # a, b 양수 보장
        if self.amp_max is None:
            a_x = F.softplus(a_x) + eps;  a_e = F.softplus(a_e) + eps
            b_x = F.softplus(b_x) + eps;  b_e = F.softplus(b_e) + eps
        else:
            sig = torch.sigmoid
            a_x = sig(a_x) * self.amp_max + eps;  a_e = sig(a_e) * self.amp_max + eps
            b_x = sig(b_x) * self.amp_max + eps;  b_e = sig(b_e) * self.amp_max + eps

        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {
            'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
            'p1_x': p1_x, 'p1_e': p1_e, 'p2_x': p2_x, 'p2_e': p2_e,
            'a_x': a_x, 'a_e': a_e, 'b_x': b_x, 'b_e': b_e,
            'kappa_x': kappa_x, 'kappa_e': kappa_e
        }

    # ----- core formulas -----
    def L(self, log_y, y, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p1  = params['p1_x']  if side=='x' else params['p1_e']
        p2  = params['p2_x']  if side=='x' else params['p2_e']
        a   = params['a_x']   if side=='x' else params['a_e']
        b   = params['b_x']   if side=='x' else params['b_e']
        # y**p는 exp(p*log y)로 계산하면 안정적
        term1 = a * torch.exp(p1 * log_y)
        term2 = b * torch.exp(p2 * log_y)
        return (1.0 - tau) * term1 + tau * term2

    def g(self, y, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        p1  = params['p1_x']  if side=='x' else params['p1_e']
        p2  = params['p2_x']  if side=='x' else params['p2_e']
        a   = params['a_x']   if side=='x' else params['a_e']
        b   = params['b_x']   if side=='x' else params['b_e']
        # y^{p}는 exp(p*log y)로
        logy = torch.log(y)
        y_p1 = torch.exp(p1 * logy)
        y_p2 = torch.exp(p2 * logy)
        denom = (1.0 - tau) * a * p1 * y_p1 + tau * b * p2 * y_p2
        return y / denom
