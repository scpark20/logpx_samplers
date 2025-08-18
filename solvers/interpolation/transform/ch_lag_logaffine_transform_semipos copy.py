import torch
import torch.nn.functional as F
from .transform import Transform

class LogAffineTransform(Transform):
    def __init__(self,
        gamma_push=False,
        gamma_max=None,
        tau_offset=0,
        tau_max=None,
        eps=1e-2,
        tau_tol=1e-6):
        self.gamma_push = gamma_push
        self.gamma_max = gamma_max
        self.tau_offset = tau_offset
        self.tau_max = tau_max
        self.eps = eps
        self.tau_tol = tau_tol
        
    def unpack(self, params):
        # params: [B, ?]  (gamma, tau_x, tau_e가 앞 3개라고 가정)
        gamma = params[:, 0]
        tau_x = params[:, 1] + self.tau_offset
        tau_e = params[:, 2] + self.tau_offset

        if self.gamma_max is not None:
            gamma = torch.tanh(gamma) * self.gamma_max
        if self.tau_max is None:
            # [0, +inf): softplus  (학습 초반엔 작고, 점차 커질 수 있음)
            tau_x = F.softplus(tau_x)
            tau_e = F.softplus(tau_e)
        else:
            # [0, tau_max): sigmoid로 상한도 함께 관리
            tau_x = self.tau_max * torch.sigmoid(tau_x)
            tau_e = self.tau_max * torch.sigmoid(tau_e)
        if self.gamma_push:
            gamma = self.push_away(gamma,  1, self.eps)
            gamma = self.push_away(gamma, -1, self.eps)

        return {'gamma': gamma, 'tau_x': tau_x, 'tau_e': tau_e}

    def L(self, y, p, side='x'):
        tau = (p['tau_x'] if side=='x' else p['tau_e'])[:, None]
        u_lin    = y
        u_nonlin = torch.log1p(tau * y) / tau   # τ>0에서 안정
        return torch.where(tau.abs() < self.tau_tol, u_lin, u_nonlin)

    # -------------------- ∫ e^{τu} u^k du (k=0..K-1), 배치 --------------------
    @staticmethod
    def _exp_poly_integrals(uc, un, tau, K, tau_tol=1e-6):
        """
        I_k = ∫_{uc}^{un} e^{τu} u^k du,  k=0..K-1
        τ≈0: (un^{k+1}-uc^{k+1})/(k+1)
        τ≠0: 부분적분 재귀  J(u,k) = e^{τu}u^k/τ - (k/τ)J(u,k-1)
        """
        B = uc.shape[0]
        uc = uc.view(B, 1); un = un.view(B, 1); tau = tau.view(B, 1)

        # (A) τ≈0 다항 적분(정확)
        ks = torch.arange(1, K + 1, device=uc.device, dtype=uc.dtype).unsqueeze(0)  # (1,K)
        I_poly = (un.pow(ks) - uc.pow(ks)) / ks                                     # (B,K)

        # (B) τ≠0 재귀 (큰 |τ|만 부분계산하여 수치 안전)
        big = (tau.abs() >= tau_tol).squeeze(1)  # (B,)

        I_rec = torch.zeros(B, K, dtype=uc.dtype, device=uc.device)
        if big.any():
            ub = un[big]; cb = uc[big]; tb = tau[big]
            exp_ub = torch.exp(tb * ub)
            exp_cb = torch.exp(tb * cb)
            I0 = (torch.expm1(tb * ub) - torch.expm1(tb * cb)) / tb  # (Bb,1)
            rows = [I0]
            J_ub = exp_ub / tb
            J_cb = exp_cb / tb
            for k in range(1, K):
                k_over_tb = (float(k)) / tb
                J_ub = (ub.pow(k) * exp_ub) / tb - k_over_tb * J_ub
                J_cb = (cb.pow(k) * exp_cb) / tb - k_over_tb * J_cb
                rows.append(J_ub - J_cb)
            I_rec[big] = torch.cat(rows, dim=1)

        # (C) 스위칭 (둘 다 미분 가능)
        return torch.where((tau.abs() < tau_tol).expand_as(I_poly), I_poly, I_rec)


    # -------------------- 라그랑주 적분 벡터 ℓ = [I_0..I_{M-1}] --------------------
    def get_integral(self, uc, un, us, tau):
        """
        반환 ℓ: (B, M),  I_k = ∫_{uc}^{un} e^{τu} u^k du
        us는 길이 M(차수)만 알려주는데 사용.
        """
        _, M = us.shape
        ell = self._exp_poly_integrals(uc, un, tau, K=M, tau_tol=1e-6)  # (B,M)
        return ell
        
    # -------------------- 반더몬드 행렬 M (B,M,M), M_{j,k}=u_j^k --------------------
    @staticmethod
    def vandermonde(us):
        """
        us: (B, M)  ->  M: (B, M, M),  열 k가 u^k (k=0..M-1, increasing)
        torch.vander(1D) 대신 배치 벡터화 버전.
        """
        B, M = us.shape
        k = torch.arange(M, device=us.device, dtype=us.dtype)  # (M,)
        return us.unsqueeze(-1).pow(k)                          # (B,M,M)

    # -------------------- 가중치 W = (M^{-1})^T ℓ  (즉, M^T W = ℓ) --------------------
    def get_coefficients(self, uc, un, us, p, side='x'):
        """
        predictor: us=[u_i, u_{i-1}, ...]
        corrector: us=[u_{i+1}, u_i, ...]  (총 노드가 predictor보다 1개 많음)
        반환 W: (B, M) — 노드값 가중치 (이걸로 xs_win, es_win과 곱-합)
        """
        tau = p['tau_x'] if side == 'x' else p['tau_e']
        ell = self.get_integral(uc, un, us, tau)          # (B,M)
        M   = self.vandermonde(us)                               # (B,M,M)
        A   = M.transpose(1, 2)                                   # M^T, (B,M,M)
        rhs = ell.unsqueeze(-1)                                   # (B,M,1)
        W = torch.linalg.lstsq(A, rhs).solution.squeeze(-1)
        return W