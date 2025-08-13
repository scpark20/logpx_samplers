import torch
import torch.nn.functional as F
from .transform import Transform

def _sigmoid_range(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)

class LogBoxCoxBlendTransform(Transform):
    """
    L = (1-τ) q log y + τ * (y^λ - 1)/λ
    g = y / ((1-τ) q + τ y^λ)
    ※ 추가적인 safe_div/클램프 없이 파라미터 범위만으로 안정화
    """
    def __init__(self,
                 gamma_max=3.0, kappa_max=3.0,
                 lam_bounds=(0.2, 3.0),   # λ ∈ (0.2, 3.0)
                 q_bounds=(0.1, 4.0),     # q ∈ (0.1, 4.0)
                 tau_min=1e-3,            # τ ∈ [tau_min, 1 - tau_min]
                 gamma_push=True, eps=1e-2):
        self.gamma_push = gamma_push
        self.gamma_max  = gamma_max
        self.kappa_max  = kappa_max
        self.lam_lo, self.lam_hi = lam_bounds
        self.q_lo,   self.q_hi   = q_bounds
        self.tau_min = tau_min
        self.eps = eps

    def unpack(self, params):
        # [B, 9] = [gamma, tau_x, tau_e, lam_x, lam_e, q_x, q_e, kappa_x, kappa_e]
        gamma, tau_x, tau_e, lam_x, lam_e, q_x, q_e, kappa_x, kappa_e = \
            [params[:, i] for i in range(9)]

        # 파라미터를 부드럽게 유한구간으로 매핑
        tau_x = _sigmoid_range(tau_x, self.tau_min, 1.0 - self.tau_min)
        tau_e = _sigmoid_range(tau_e, self.tau_min, 1.0 - self.tau_min)
        lam_x = _sigmoid_range(lam_x, self.lam_lo, self.lam_hi)
        lam_e = _sigmoid_range(lam_e, self.lam_lo, self.lam_hi)
        q_x   = _sigmoid_range(q_x,   self.q_lo,   self.q_hi)
        q_e   = _sigmoid_range(q_e,   self.q_lo,   self.q_hi)

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.kappa_max is not None:
            kappa_x = torch.tanh(kappa_x) * self.kappa_max
            kappa_e = torch.tanh(kappa_e) * self.kappa_max

        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {
            'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e,
            'lam_x': lam_x, 'lam_e': lam_e, 'q_x': q_x, 'q_e': q_e,
            'kappa_x': kappa_x, 'kappa_e': kappa_e
        }

    def L(self, log_y, y, params, side='x'):
        # L = (1-τ) q log y + τ * (y^λ - 1)/λ
        tau = params['tau_x'] if side=='x' else params['tau_e']
        lam = params['lam_x'] if side=='x' else params['lam_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        a = lam * log_y
        boxcox = torch.expm1(a) / lam        # λ→0에서도 자연스럽게 수렴
        return (1.0 - tau) * q * log_y + tau * boxcox

    def g(self, y, params, side='x'):
        # g = y / ((1-τ) q + τ y^λ)  (분모 추가 안정화 없음)
        tau = params['tau_x'] if side=='x' else params['tau_e']
        lam = params['lam_x'] if side=='x' else params['lam_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        # y는 보통 exp(log_y)라 >0 가정. 그 전제하에 log/clamp 생략.
        y_pow = torch.exp(lam * torch.log(y))
        denom = (1.0 - tau) * q + tau * y_pow
        return y / denom

    # 선택: Δu를 로그 도메인에서 안정적으로 (L(yn)-L(yc) 대신 쓰면 좋음)
    def delta_u_from_logs(self, log_yc, log_yn, params, side='x'):
        tau = params['tau_x'] if side=='x' else params['tau_e']
        lam = params['lam_x'] if side=='x' else params['lam_e']
        q   = params['q_x']   if side=='x' else params['q_e']
        dlog = log_yn - log_yc
        a_n, a_c = lam * log_yn, lam * log_yc
        m = torch.maximum(a_n, a_c)
        delta_boxcox = torch.exp(m) * (torch.expm1(a_n - m) - torch.expm1(a_c - m)) / lam
        return (1.0 - tau) * q * dlog + tau * delta_boxcox
