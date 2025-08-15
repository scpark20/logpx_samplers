# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import pickle
import numpy as np

from .solver import Solver

class DS_Solver(Solver):
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
        learned_params_path=None,
        default_steps=10,
        device=None,
        enable_training=False
    ):
        super().__init__(model_fn, noise_schedule, algorithm_type)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_training = enable_training

        if learned_params_path and os.path.exists(learned_params_path):
            self._load_learned_params(learned_params_path)
        else:
            self._initialize_default_params(default_steps)

        self._timesteps_cache = None
        self._solver_matrix_cache = None

        if self.enable_training:
            self._setup_training()

    def _initialize_default_params(self, steps):
        self.steps = steps
        self.r_params = nn.Parameter(torch.ones(steps, device=self.device))
        self.c_matrix = nn.Parameter(torch.zeros(steps, steps - 1, device=self.device))
        print(f"DS-Solver: Initialized with default Euler-like parameters for {steps} steps.")

    def _load_learned_params(self, path):
        print(f"DS-Solver: Loading learned parameters from {path}")
        if path.endswith('.json'):
            with open(path, 'r') as f: data = json.load(f)
        else:
            with open(path, 'rb') as f: data = pickle.load(f) if path.endswith('.pkl') else torch.load(f)
        params = data.get('parameters', data)
        
        self.steps = len(params.get('timesteps', [0] * (10 + 1))) -1
        self.r_params = nn.Parameter(torch.tensor(params.get('r_params', np.ones(self.steps)), device=self.device, dtype=torch.float32))
        self.c_matrix = nn.Parameter(torch.zeros(self.steps, self.steps - 1, device=self.device))
        if 'c_params' in params:
            for i, c_param in enumerate(params['c_params']):
                if i + 1 < self.steps:
                    c_tensor = torch.tensor(c_param, device=self.device, dtype=torch.float32)
                    self.c_matrix.data[i + 1, :len(c_tensor)] = c_tensor
        print(f"DS-Solver: Loaded parameters for {self.steps} steps.")

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
                M = torch.zeros(self.steps, self.steps, device=self.device)
                M[0, 0] = 1.0
                for i in range(1, self.steps):
                    M[i, :i] = self.c_matrix[i, :i]
                    M[i, i] = 1.0 - M[i, :i].sum()
                self._solver_matrix_cache = M
        return self._solver_matrix_cache

    def invalidate_cache(self):
        self._timesteps_cache = None
        self._solver_matrix_cache = None

    def sample(self, x, steps=None, **kwargs):
        if steps is not None and steps != self.steps:
            print(f"Warning: DS-Solver was trained for {self.steps} steps, but {steps} requested. Using {self.steps}.")

        timesteps = self.get_timesteps()
        solver_matrix = self.get_solver_matrix()
        
        velocity_buffer = []
        x_t = x
        
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM_DISABLE", "true")):
            t_current, t_next = timesteps[i], timesteps[i+1]
            v_current = self.model(x_t, t_current)
            velocity_buffer.append(v_current)
            
            v_stack = torch.stack(velocity_buffer)
            weights = solver_matrix[i, :i+1]
            
            v_combined = torch.einsum('n,n...->...', weights.to(v_stack.dtype), v_stack)
            
            dt = t_next - t_current
            x_t = x_t + v_combined * dt
        return x_t

    def _setup_training(self):
        try:
            from lion_pytorch import Lion
            self.optimizer = Lion(self.parameters(), lr=0.001, weight_decay=0.0)
            print("DS-Solver: Using Lion optimizer for training.")
        except ImportError:
            raise ImportError("Lion optimizer not found. Please install it by running: pip install lion-pytorch")

    def parameters(self):
        return [self.r_params, self.c_matrix]
    
    def _get_learned_trajectory(self, x_batch):
        self.invalidate_cache()
        time_deltas = F.softmax(self.r_params, dim=0)
        timesteps = torch.cumsum(time_deltas, dim=0)
        timesteps = torch.cat([torch.zeros(1, device=self.device), timesteps], dim=0)
        
        M = torch.zeros(self.steps, self.steps, device=self.device)
        M[0, 0] = 1.0
        for i in range(1, self.steps):
            M[i, :i] = self.c_matrix[i, :i]
            M[i, i] = 1.0 - M[i, :i].sum()
        solver_matrix = M

        trajectory = [x_batch.clone()]
        velocity_buffer = []
        x_t = x_batch
        
        for i in range(self.steps):
            t_current, t_next = timesteps[i], timesteps[i+1]
            v_t = self.model(x_t, t_current)
            velocity_buffer.append(v_t)
            v_stack = torch.stack(velocity_buffer)
            weights = solver_matrix[i, :i+1]
            v_combined = torch.einsum('n,n...->...', weights.to(v_stack.dtype), v_stack)
            dt = t_next - t_current
            x_t = x_t + v_combined * dt
            trajectory.append(x_t.clone())
            
        return timesteps, trajectory

    @torch.no_grad()
    def _generate_reference_trajectory(self, x_batch, reference_steps):
        ref_timesteps = torch.linspace(0, 1, reference_steps + 1, device=self.device)
        ref_trajectory = [x_batch.clone()]
        x_t = x_batch
        for i in range(reference_steps):
            v_t = self.model(x_t, ref_timesteps[i])
            dt = ref_timesteps[i+1] - ref_timesteps[i]
            x_t = x_t + v_t * dt
            ref_trajectory.append(x_t.clone())
        return ref_timesteps, ref_trajectory

    def compute_loss(self, x_batch, ref_timesteps, ref_trajectory):
        learned_timesteps, learned_trajectory = self._get_learned_trajectory(x_batch)
        total_mse_loss = 0.0
        for i in range(1, len(learned_trajectory)):
            current_t = learned_timesteps[i]
            ref_idx = torch.argmin(torch.abs(ref_timesteps - current_t)).item()
            learned_state, ref_state = learned_trajectory[i], ref_trajectory[ref_idx]
            total_mse_loss += F.mse_loss(learned_state, ref_state)

        mse_loss = total_mse_loss / self.steps
        huber_loss = F.smooth_l1_loss(learned_trajectory[-1], ref_trajectory[-1])
        
        return mse_loss + huber_loss
    
    def train(self, num_samples, batch_size, reference_steps, save_path, input_shape):
        if not self.enable_training:
            raise RuntimeError("Training not enabled. Initialize with enable_training=True")
        
        num_iterations = max(1, num_samples // batch_size)
        print(f"\n{'='*60}\nDS-Solver Training Started\n{'='*60}")
        pbar = tqdm(range(num_iterations), desc="Training DS-Solver")
        
        for iteration in pbar:
            x_batch = torch.randn(batch_size, *input_shape, device=self.device, dtype=torch.float32)
            ref_timesteps, ref_trajectory = self._generate_reference_trajectory(x_batch, reference_steps)
            
            self.optimizer.zero_grad()
            
            # <<< 수정된 부분: 수치 안정성을 위해 AMP(autocast)를 사용하지 않고 float32로만 학습합니다. >>>
            loss = self.compute_loss(x_batch, ref_timesteps, ref_trajectory)
            
            if torch.isnan(loss):
                # <<< 수정된 부분: break 대신 RuntimeError를 발생시켜 프로그램을 완전히 중지시킵니다. >>>
                raise RuntimeError(f"Loss is NaN at iteration {iteration + 1}. Stopping training.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if (iteration + 1) % 5 == 0:
                print(f"\n  [Training Monitor] Iteration {iteration + 1}/{num_iterations} -> Loss: {loss.item():.4f}")

        if not torch.isnan(loss):
             print(f"\nTraining completed! Final loss: {loss.item():.4f}")
             if save_path:
                 self.save_learned_params(save_path)
        else:
             print(f"\nTraining stopped due to NaN loss.")

    def save_learned_params(self, path):
        params_to_save = {
            'parameters': {
                'timesteps': self.get_timesteps().detach().cpu().numpy(),
                'solver_matrix': self.get_solver_matrix().detach().cpu().numpy(),
                'r_params': self.r_params.detach().cpu().numpy(),
                'c_params': [self.c_matrix[i, :i].detach().cpu().numpy() for i in range(1, self.steps)]
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(params_to_save, f)
        print(f"Learned parameters saved to {path}")
    
    def print_learned_params(self):
        print("\n--- Loaded DS-Solver Parameters ---")
        timesteps = self.get_timesteps().cpu().numpy()
        print(f"Learned Timesteps ({len(timesteps)-1} steps):")
        for i, t in enumerate(timesteps):
            print(f"  t_{i}: {t:.4f}")