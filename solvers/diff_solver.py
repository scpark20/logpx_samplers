import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle

from .solver import Solver

# 유틸: no-op context
class nullcontext:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

# =================================================================
# DS_Solver: 샘플링(추론) 전용 클래스
# =================================================================
class DS_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
        learned_params_path=None,
        steps=10,
        use_amp_sample=True,
        amp_dtype="bf16",
    ):
        super().__init__(model_fn, noise_schedule, algorithm_type)
        assert self.algorithm_type in ["data_prediction", "noise_prediction", "vector_prediction"], \
            "DS_Solver supports 'data_prediction', 'noise_prediction', or 'vector_prediction'"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if learned_params_path and os.path.exists(learned_params_path):
            self._load_learned_params(learned_params_path)
        else:
            self._initialize_default_params(steps)

        self._timesteps_cache = None
        self._solver_matrix_cache = None

        self.use_amp_sample = use_amp_sample and (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16

    def _initialize_default_params(self, steps):
        self.steps = steps
        self.r_params = nn.Parameter(torch.ones(steps, device=self.device), requires_grad=True)
        self.c_matrix = nn.Parameter(torch.zeros(steps, steps - 1, device=self.device), requires_grad=True)

    def _load_learned_params(self, path):
        print(f"DS-Solver: Loading learned parameters from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        params = data.get('parameters', data)
        
        self.steps = len(params.get('timesteps', [0] * (10 + 1))) - 1
        self.r_params = nn.Parameter(torch.tensor(params['r_params'], device=self.device, dtype=torch.float32), requires_grad=False)
        self.c_matrix = nn.Parameter(torch.zeros(self.steps, self.steps - 1, device=self.device), requires_grad=False)
        if 'c_params' in params:
            for i, c_param in enumerate(params['c_params']):
                if i + 1 < self.steps:
                    c_tensor = torch.tensor(c_param, device=self.device, dtype=torch.float32)
                    self.c_matrix.data[i + 1, :len(c_tensor)] = c_tensor
        print(f"DS-Solver: Loaded parameters for {self.steps} steps.")

    def parameters(self):
        return [self.r_params, self.c_matrix]

    def get_timesteps(self):
        if self._timesteps_cache is None:
            with torch.no_grad():
                time_deltas = F.softmax(self.r_params, dim=0)
                timesteps = torch.cumsum(time_deltas, dim=0)
                self._timesteps_cache = torch.cat([torch.zeros(1, device=self.device), timesteps], dim=0)
        return self._timesteps_cache

    def get_solver_matrix(self):
        if self._solver_matrix_cache is None:
            with torch.no_grad():
                M = torch.zeros(self.steps, self.steps, device=self.device); M[0, 0] = 1.0
                for i in range(1, self.steps):
                    M[i, :i] = self.c_matrix[i, :i]; M[i, i] = 1.0 - M[i, :i].sum()
                self._solver_matrix_cache = M
        return self._solver_matrix_cache
        
    def invalidate_cache(self):
        self._timesteps_cache = None
        self._solver_matrix_cache = None

    def scale_timesteps(self, timesteps_in_0_1):
        t_0 = 1. / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        return t_0 + timesteps_in_0_1 * (t_T - t_0 - 1e-4)

    # DDPM/VP: x0 결합식
    def _ds_update_step(self, x, s, t, combined_x0):
        alpha_s, sigma_s = self.noise_schedule.marginal_alpha(s), self.noise_schedule.marginal_std(s)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        omega_s = alpha_s / (sigma_s + 1e-8)
        omega_t = alpha_t / (sigma_t + 1e-8)
        return (sigma_t / sigma_s) * x + sigma_t * (omega_t - omega_s) * combined_x0

    # DDPM/VP: 노이즈 결합식
    def _dpm_noise_update_step(self, x_t, t_current, t_next, noise_pred):
        alpha_s, alpha_t = self.noise_schedule.marginal_alpha(t_current), self.noise_schedule.marginal_alpha(t_next)
        sigma_s, sigma_t = self.noise_schedule.marginal_std(t_current), self.noise_schedule.marginal_std(t_next)
        return (alpha_t / alpha_s) * x_t - (sigma_t * (alpha_s / alpha_t) - sigma_s) * noise_pred

    # Rectified Flow(velocity)
    def _rf_velocity_update_step(self, x, s, t, combined_v):
        dt = (t - s)
        return x + dt * combined_v

    @torch.no_grad()
    def sample(self, x, steps=None, **kwargs):
        if steps is not None and steps != self.steps:
            print(f"Warning: DS-Solver trained for {self.steps} steps, but {steps} requested.")
        
        base_timesteps = self.get_timesteps()
        scaled_timesteps = self.scale_timesteps(base_timesteps)
        timesteps = torch.flip(scaled_timesteps, [0])
        solver_matrix = self.get_solver_matrix()
        
        x_t = x
        out_buf = torch.empty((self.steps,) + x.shape, device=x.device, dtype=x.dtype)

        amp_ctx = torch.amp.autocast('cuda',dtype=self.amp_dtype) if self.use_amp_sample and x.is_cuda else nullcontext()

        with amp_ctx:
            for i in tqdm(range(self.steps), disable=os.getenv("TQDM_DISABLE", "true")):
                t_current, t_next = timesteps[i], timesteps[i+1]
                model_output = self.model_fn(x_t, t_current)
                # dtype 정합
                if model_output.dtype != x_t.dtype:
                    model_output = model_output.to(x_t.dtype)
                out_buf[i].copy_(model_output)

                weights = solver_matrix[i, :i+1].to(out_buf.dtype).view(i+1, *([1] * (x_t.ndim)))
                combined_output = (out_buf[:i+1] * weights).sum(dim=0)
                
                if self.algorithm_type == "data_prediction":
                    x_t = self._ds_update_step(x_t, t_current, t_next, combined_output)
                elif self.algorithm_type == "noise_prediction":
                    x_t = self._dpm_noise_update_step(x_t, t_current, t_next, combined_output)
                else:  # vector_prediction (RF)
                    x_t = self._rf_velocity_update_step(x_t, t_current, t_next, combined_output)
        return x_t
    
    def print_learned_params(self):
        print("\n--- Loaded DS-Solver Parameters ---")
        timesteps = self.get_timesteps().cpu().numpy()
        print(f"Learned Timesteps ({len(timesteps)-1} steps):")
        for i, t in enumerate(timesteps):
            print(f"  t_{i}: {t:.4f}")

# =================================================================
# DSTrainer: DS_Solver 학습 전용 클래스
# =================================================================
class DSTrainer:
    def __init__(
        self, 
        model, 
        algorithm_type="data_prediction", 
        target_steps=10, 
        reference_steps=10,
        use_amp=True,
        amp_dtype="bf16",
        debug=False
    ):
        self.model = model
        self.algorithm_type = algorithm_type
        self.target_steps = target_steps
        self.reference_steps = reference_steps
        self.device = model.device
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
        self.debug = debug
        self._init_trainer()

    def _init_trainer(self):
        dummy_conds = [0] if type(self.model).__name__ == 'DiT' else [""]
        # 검색은 항상 단일 패스(no CFG)로: guidance_scale=1.0
        train_model_fn, train_noise_schedule, _ = self.model.get_model_fn(
            pos_conds=dummy_conds, guidance_scale=1.0, seeds=[42]
        )

        # 모델 파라미터를 완전히 고정
        try:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        except Exception:
            pass

        # no_grad 보장 래퍼 (일부 백본이 내부에서 enable_grad를 시도하더라도 외부에서 no_grad)
        def _model_fn_nograd(x, t):
            with torch.no_grad():
                return train_model_fn(x, t)

        self.solver = DS_Solver(
            _model_fn_nograd, train_noise_schedule, self.algorithm_type, 
            learned_params_path=None, steps=self.target_steps,
            use_amp_sample=self.use_amp,
            amp_dtype="bf16" if self.amp_dtype == torch.bfloat16 else "fp16",
        )

    def train(self, num_samples, batch_size, save_path, lr=0.01, weight_decay=0.0):
        try:
            from lion_pytorch import Lion
            optimizer = Lion(self.solver.parameters(), lr=lr, weight_decay=weight_decay)
            print(f"DSTrainer: Using Lion optimizer with lr={lr} to train BOTH r_params and c_matrix.")
        except ImportError:
            raise ImportError("Lion optimizer not found. Please install it by running: pip install lion-pytorch")
        
        num_iterations = max(1, num_samples // batch_size)
        pbar = tqdm(range(num_iterations), desc="Training DS-Solver")
        last_loss = None
        
        for iteration in pbar:
            C = self.model.pipe.transformer.config.in_channels
            S = self.model.pipe.transformer.config.sample_size
            x_batch = torch.randn(
                batch_size, C, S, S, device=self.device, dtype=torch.float32
            )
            ref_trajectory = self._generate_reference_trajectory(x_batch, self.reference_steps)

            optimizer.zero_grad(set_to_none=True)
            loss = self._compute_loss(x_batch, ref_trajectory, iteration)
            if torch.isnan(loss):
                print(f"\n!!! Loss is NaN at iteration {iteration + 1}. Halting training. !!!")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.solver.parameters(), 1.0)
            optimizer.step()
            last_loss = loss.item()
            pbar.set_postfix({'loss': f'{last_loss:.4f}'})
        
        if last_loss is not None and not (last_loss != last_loss):
            print(f"\nTraining completed! Final loss: {last_loss:.4f}")
            self._save_learned_params(save_path)

    def _get_learned_trajectory(self, x_batch, iteration):
        # r/c가 변하므로 캐시 무효화
        self.solver.invalidate_cache()

        # 학습 가능한 타임스텝
        time_deltas = F.softmax(self.solver.r_params, dim=0)
        base_timesteps_0_1 = torch.cat(
            [torch.zeros(1, device=self.device), torch.cumsum(time_deltas, dim=0)],
            dim=0
        )
        timesteps = self.solver.scale_timesteps(base_timesteps_0_1)
        timesteps = torch.flip(timesteps, [0])  # 역방향

        trajectory = [x_batch.clone()]
        x_t = x_batch

        # 과거 모델 출력들을 보관하는 리스트(그래프에 엮지 않음)
        saved_outputs = []  # 각 원소 shape: [B, C, H, W], dtype=x_t.dtype, requires_grad=False

        amp_ctx = (
            torch.amp.autocast('cuda', dtype=self.amp_dtype)
            if self.use_amp and x_batch.is_cuda else nullcontext()
        )

        for i in range(self.solver.steps):
            t_current, t_next = timesteps[i], timesteps[i+1]

            # 모델 추론: 항상 no_grad (모델은 상수)
            with amp_ctx, torch.no_grad():
                model_output = self.solver.model_fn(x_t, t_current)
                if model_output.dtype != x_t.dtype:
                    model_output = model_output.to(x_t.dtype)

            # 리스트에 고정 복사본으로 보관(그래프에 엮이지 않음)
            saved_outputs.append(model_output.detach())

            # 가중치 벡터: [c_row, diag_w]
            c_row = self.solver.c_matrix[i, :i]            # [i] (파라미터 뷰)
            diag_w = 1.0 - c_row.sum()                     # 스칼라
            weights = torch.cat([c_row, diag_w.view(1)], dim=0)  # [i+1]

            # 과거 출력 스택(새 텐서) + 브로드캐스트 가중합
            output_stack = torch.stack(saved_outputs, dim=0)  # [i+1, B, C, H, W]
            w_b = weights.to(output_stack.dtype).view(i+1, *([1] * (x_t.ndim)))
            combined_output = (output_stack * w_b).sum(dim=0)

            # 업데이트(여기서만 grad 필요)
            if self.solver.algorithm_type == "data_prediction":
                x_t = self.solver._ds_update_step(x_t, t_current, t_next, combined_output)
            elif self.solver.algorithm_type == "noise_prediction":
                x_t = self.solver._dpm_noise_update_step(x_t, t_current, t_next, combined_output)
            else:  # vector_prediction (Rectified Flow)
                x_t = self.solver._rf_velocity_update_step(x_t, t_current, t_next, combined_output)

            if torch.isnan(x_t).any():
                if self.debug:
                    print(f"!!! NaN detected in x_t after update at Step {i} in Iteration {iteration + 1} !!!")
                break

            trajectory.append(x_t.clone())

        return timesteps, trajectory

    @torch.no_grad()
    def _generate_reference_trajectory(self, x_batch, reference_steps):
        NS = self.solver.noise_schedule
        skip = 'time_uniform_flow' if self.algorithm_type == 'vector_prediction' else 'time_uniform'

        # T → 0 방향으로 N+1개 그리드 생성
        timesteps = self.solver.get_time_steps(
            skip_type=skip,
            t_T=NS.T,
            t_0=1.0 / NS.total_N,
            N=reference_steps,
            device=self.device,
            shift=1.0
        )

        trajectory = [x_batch.clone()]
        x_t = x_batch

        for i in range(reference_steps):
            t_current, t_next = timesteps[i], timesteps[i + 1]
            model_output = self.solver.model_fn(x_t, t_current)

            if self.solver.algorithm_type == "vector_prediction":
                # Rectified Flow: Euler
                x_t = x_t + (t_next - t_current) * model_output
            else:
                # DDPM/VP (논문식 Euler-일계)
                lambda_current = NS.marginal_lambda(t_current)
                lambda_next    = NS.marginal_lambda(t_next)
                h = lambda_next - lambda_current

                sigma_current  = NS.marginal_std(t_current)
                sigma_next     = NS.marginal_std(t_next)
                signal_rate_next = NS.marginal_alpha(t_next)

                x_t = (sigma_next / sigma_current) * x_t \
                    + (-signal_rate_next * torch.expm1(-h)) * model_output

            trajectory.append(x_t.clone())

        return trajectory
    
    def _compute_loss(self, x_batch, ref_trajectory, iteration):
        learned_timesteps, learned_trajectory = self._get_learned_trajectory(x_batch, iteration)
        
        if len(learned_trajectory) <= 1 or (learned_trajectory[-1]).isnan().any():
            return torch.tensor(float('nan'), device=self.device)

        total_mse_loss = 0.0
        ref_len = len(ref_trajectory)
        
        learned_timesteps_forward = torch.flip(learned_timesteps, [0])
        
        for i in range(1, len(learned_trajectory)):
            learned_state = learned_trajectory[i]
            current_t = learned_timesteps_forward[i]
            t_0 = 1. / self.solver.noise_schedule.total_N; t_T = self.solver.noise_schedule.T
            ratio = (current_t - t_0) / (t_T - t_0)
            ref_idx = min(ref_len - 1, int(ratio.item() * (ref_len - 1)))
            ref_state = ref_trajectory[ref_idx].expand_as(learned_state)
            total_mse_loss = total_mse_loss + F.mse_loss(learned_state, ref_state)
            
        mse_loss = total_mse_loss / self.solver.steps
        huber_loss = F.smooth_l1_loss(learned_trajectory[-1], ref_trajectory[-1].expand_as(learned_trajectory[-1]))
        return mse_loss + huber_loss

    def _save_learned_params(self, path):
        with torch.no_grad():
            M = torch.zeros(self.solver.steps, self.solver.steps, device=self.device)
            M[0, 0] = 1.0
            for i in range(1, self.solver.steps):
                row = self.solver.c_matrix[i, :i]
                M[i, :i] = row
                M[i, i] = 1.0 - row.sum()

            params_to_save = {
                'parameters': {
                    'timesteps': self.solver.get_timesteps().detach().cpu().numpy(),
                    'solver_matrix': M.detach().cpu().numpy(),
                    'r_params': self.solver.r_params.detach().cpu().numpy(),
                    'c_params': [self.solver.c_matrix[i, :i].detach().cpu().numpy() for i in range(1, self.solver.steps)]
                }
            }
        with open(path, 'wb') as f:
            pickle.dump(params_to_save, f)
        print(f"Learned parameters saved to {path}")