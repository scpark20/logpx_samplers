# neural_solver.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm

from ...solver import Solver  # 베이스 클래스(질문에 준 코드)와 동일한 import 패턴

class NeuralSolver(Solver):
    """
    Differentiable Solver Search (DS) 스타일의 범용 뉴럴 솔버.
    - 학습 파라미터:
        * self.log_deltas     : softmax -> Δt_i >= 0, ΣΔt_i = (t_hi - t_lo)
        * self.raw_coeffs     : 엄격 하삼각 파라미터; i번째 행의 j<i에 대해 자유변수,
                               대각 성분은 1 - Σ_{j<i} c_{ij} 로 자동 완성 (행합=1 보장)
    - 업데이트:
        * (FM/ODE)   x_{i+1} = x_i + Δt_i · Σ_{j=0}^i M_{ij} · v_j
          (여기서 v_j = model_fn(x_j, t_j)의 출력; algorithm_type='vector_prediction' 권장)
        * (VP/DDPM)  확장 훅 포함(주석). λ=α/σ 기반 업데이트 계수/운반항 분리 시 구현.
    """
    def __init__(
        self,
        noise_schedule,
        steps: int,
        algorithm_type: str = "vector_prediction",  # 'vector_prediction' 권장(FM/ODE)
        skip_type: str = "time_uniform_flow",
        flow_shift: float = 1.0,
        time_learning: bool = True,
        coeff_transfer: str = "none",   # {'none','tanh'}
        coeff_clip: float | None = None,
        small_nfe_stabilize: bool = True,  # NFE<=6에서 마지막 행 축소 휴리스틱
        train_mode: bool = False,
        checkpoint: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__(noise_schedule, algorithm_type)
        assert steps >= 1
        self.steps               = steps
        self.skip_type           = skip_type
        self.flow_shift          = flow_shift
        self.time_learning       = time_learning
        self.coeff_transfer      = coeff_transfer
        self.coeff_clip          = coeff_clip
        self.small_nfe_stabilize = small_nfe_stabilize
        self.train_mode          = train_mode
        self.checkpoint          = checkpoint
        self.eps                 = eps

        # ----- 초기 time grid에서 Δt 초기화 -----
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        # 베이스의 get_time_steps를 이용해 초기 분할(내림차순) 확보
        with torch.no_grad():
            ts0 = self.get_time_steps(
                skip_type=self.skip_type, t_T=t_T, t_0=t_0, N=steps, device="cpu", shift=self.flow_shift
            ).to(torch.float64)  # num 안정 위해 float64
            # Δt_i = ts[i] - ts[i+1] (ts는 내림차순이라 Δt_i >= 0)
            init_deltas = ts0[:-1] - ts0[1:]
            init_deltas = torch.clamp(init_deltas, min=self.eps)
        self.log_deltas = nn.Parameter(init_deltas.log().to(torch.float32))  # (S,)

        # ----- 계수 파라미터 (strict lower-triangular 자유변수) -----
        # raw_coeffs[i, j]는 j<i에서만 사용됨. 대각은 1 - sum(prev)로 채움.
        self.raw_coeffs = nn.Parameter(torch.zeros(steps, steps, dtype=torch.float32))

        # 사전계산/추론용 테이블 모드(선택)
        self.register_buffer("_table_mode", torch.tensor(0, dtype=torch.uint8), persistent=False)
        self.register_buffer("_table_deltas", torch.empty(0), persistent=False)
        self.register_buffer("_table_coeffs", torch.empty(0), persistent=False)

    # ========== 내부 유틸 ==========

    def _apply_coeff_transfer(self, row_free: torch.Tensor) -> torch.Tensor:
        """
        행의 자유변수 벡터(row_free, shape: (i,))에 안정화 변환 적용.
        """
        if self.coeff_transfer == "tanh":
            row_free = torch.tanh(row_free)
        if self.coeff_clip is not None:
            row_free = row_free.clamp(min=-self.coeff_clip, max=self.coeff_clip)
        return row_free

    def _row_coeffs(self, i: int, device, dtype) -> torch.Tensor:
        """
        i번째 스텝의 계수 벡터 w^{(i)} = [c_{i0},...,c_{i,i-1}, c_{ii}] (길이 i+1).
        대각은 1 - Σ_{j<i} c_{ij} 로 설정하여 행합=1을 보장.
        small_nfe_stabilize 옵션이 켜지고 steps<=6이면,
        마지막 두 행에서 j = i-1만 허용(휴리스틱)하도록 row_free를 0으로 제한.
        """
        if self._table_mode.item() == 1:
            # 테이블 모드: 미리 주입된 계수 사용
            w = self._table_coeffs[i, :i+1].to(device=device, dtype=dtype)
            return w

        # 학습 파라미터 사용
        row_free = self.raw_coeffs[i, :i].to(device=device, dtype=dtype)

        # 작은 NFE 안정화 휴리스틱: 마지막 행/직전 행에서 j=i-1만 사용
        if self.small_nfe_stabilize and self.steps <= 6 and i >= self.steps - 2:
            if i > 0:
                mask = torch.zeros_like(row_free)
                mask[-1] = 1.0
                row_free = row_free * mask

        row_free = self._apply_coeff_transfer(row_free)
        s = row_free.sum()
        diag = (torch.ones((), device=device, dtype=dtype) - s)
        w = torch.cat([row_free, diag.unsqueeze(0)], dim=0)  # (i+1,)
        return w

    def _deltas(self, device, dtype) -> torch.Tensor:
        """
        Δt 벡터(길이 S). table 모드면 테이블 사용, 아니면 softmax 기반 학습.
        베이스의 learned_timesteps를 사용해 실제 ts도 얻을 수 있음.
        """
        if self._table_mode.item() == 1:
            return self._table_deltas.to(device=device, dtype=dtype)

        # 베이스의 learned_timesteps()는 [t_hi,...,t_lo] (S+1,)
        ts = self.learned_timesteps(device=device, dtype=dtype)
        deltas = ts[:-1] - ts[1:]  # (S,)
        # 안전 클램프
        return torch.clamp(deltas, min=self.eps)

    def load_table(self, deltas: torch.Tensor, coeffs: torch.Tensor):
        """
        추론용 사전계산 테이블 주입.
        - deltas: shape (S,), 각 스텝 Δt
        - coeffs: shape (S, S+1) 또는 (S, S)에서 i행은 길이 i+1만 유효
        """
        assert deltas.dim() == 1 and deltas.shape[0] == self.steps
        assert coeffs.dim() == 2 and coeffs.shape[0] == self.steps
        self._table_deltas  = deltas.detach().clone()
        self._table_coeffs  = coeffs.detach().clone()
        self._table_mode[:] = 1  # enable

    def disable_table(self):
        self._table_mode[:] = 0

    # ========== 샘플링 루프 ==========

    def sample(self, x, model_fn, inter_return: bool = False, **kwargs):
        """
        FM(ODE) 기본 구현:
          x_{i+1} = x_i + Δt_i · Σ_{j=0}^i M_{ij} v_j
        여기서 v_j = self.model_fn(x_j, t_j)의 출력이며,
        algorithm_type='vector_prediction'인 경우를 권장합니다.

        DDPM/VP 확장을 원하면 아래 주석 블록(★)을 참고해 가중/운반항을 정의하세요.
        """
        self.set_model_fn(model_fn)
        device, dtype = x.device, x.dtype

        # 학습/테이블에서 Δt 및 ts 얻기
        ts = self.learned_timesteps(device=device, dtype=dtype)
        if not self.time_learning and self._table_mode.item() == 0:
            ts = ts.detach()

        deltas = self._deltas(device=device, dtype=dtype)  # (S,)

        # (필요 시) 노이즈 스케줄 값
        alphas = self.noise_schedule.marginal_alpha(ts)
        sigmas = self.noise_schedule.marginal_std(ts)
        # lambdas = alphas / sigmas  # VP 확장시 사용

        # 인터미디엇 리턴 옵션
        if inter_return:
            import numpy as np
            return_index = np.random.randint(0, self.steps)
            inter = None

        use_tqdm = os.getenv("DPM_TQDM", "1") not in ("0", "False", "false", "")
        context = nullcontext() if self.train_mode else torch.no_grad()

        # 예측 캐시
        preds = []          # v_j(or pred_j)
        xs    = [x]         # x_0, x_1, ...

        with context:
            # t_0에서 첫 예측
            v0 = self.checkpoint_model_fn(x, ts[0]) if (self.train_mode and self.checkpoint) else self.model_fn(x, ts[0])
            # dual_prediction 같은 복합 출력은 사용하지 않으므로 방어
            if isinstance(v0, (tuple, list)):
                raise ValueError("NeuralSolver expects single-tensor prediction (use 'vector_prediction' or adapt model_fn).")
            preds.append(v0)
            if inter_return and return_index == 0:
                inter = v0

            for i in tqdm(range(self.steps), disable=not use_tqdm):
                # === 계수 행 구성 ===
                w_i = self._row_coeffs(i, device=device, dtype=dtype)   # (i+1,)
                # === 가중 예측 합 ===
                acc = 0
                for j in range(i + 1):
                    acc = acc + w_i[j] * preds[j]    # 안전 브로드캐스트(스칼라 * 텐서)

                # === 업데이트 ===
                # FM(ODE): x_{i+1} = x_i + Δt_i * acc
                x_next = xs[-1] + deltas[i] * acc

                # ★ (선택) DDPM/VP 확장(ω=α/σ):
                # 예: x_next = (sigmas[i+1]/sigmas[i]) * xs[-1] + sigmas[i+1] * (lambdas[i+1]-lambdas[i]) * acc
                # 정확한 계수는 사용 중인 스케줄/모델 정의에 맞추어 교체하세요.

                xs.append(x_next)

                if i < self.steps - 1:
                    v_next = self.checkpoint_model_fn(x_next, ts[i + 1]) if (self.train_mode and self.checkpoint) else self.model_fn(x_next, ts[i + 1])
                    if isinstance(v_next, (tuple, list)):
                        raise ValueError("NeuralSolver expects single-tensor prediction; got tuple/list.")
                    preds.append(v_next)
                    if inter_return and return_index == (i + 1):
                        inter = v_next

        if inter_return:
            return xs[-1], inter
        else:
            return xs[-1]

    # (선택) 학습 편의 함수
    def freeze_time(self, freeze: bool = True):
        self.log_deltas.requires_grad_(not freeze)

    def export_learned_params(self, device=None):
        """
        현재 파라미터(Δt, M 행) 텐서로 export: (deltas, coeffs)
        coeffs는 (S, S+1) 형태로 각 행의 유효 길이는 i+1.
        """
        if device is None:
            device = self.log_deltas.device
        dtype = torch.float32

        ts = self.learned_timesteps(device=device, dtype=dtype)
        deltas = (ts[:-1] - ts[1:]).detach()

        coeff_rows = []
        for i in range(self.steps):
            w_i = self._row_coeffs(i, device=device, dtype=dtype)
            # S+1 길이로 패딩 (뒤쪽은 0)
            pad_len = self.steps + 1 - (i + 1)
            if pad_len > 0:
                w_i = torch.cat([w_i, torch.zeros(pad_len, device=device, dtype=dtype)], dim=0)
            coeff_rows.append(w_i.unsqueeze(0))
        coeffs = torch.cat(coeff_rows, dim=0)  # (S, S+1)
        return deltas, coeffs
