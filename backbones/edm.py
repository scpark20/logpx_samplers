import torch
import numpy as np
from diffusers import DiTPipeline
from typing import Optional, List, Tuple, Union
from .backbone import Backbone
from PIL import Image
import pickle

class EDM(Backbone):
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "/dataset/edm/edm-cifar10-32x32-uncond-vp.pkl",
        trainable = False
    ):
        super().__init__(trainable)
        self.device = torch.device(device)
        self.dtype = dtype

        # Load and move pipeline
        with open(model_id, "rb") as f:
            self.net = pickle.load(f)["ema"].to(device)
        self.net.to(device=self.device, dtype=self.dtype)

    def set_freeze(self):
        self.net.eval()
        for p in self.net.parameters():  # 확실히 freeze
            p.requires_grad_(False)

    def prepare_noise(
        self, seeds: List[int],
    ) -> torch.Tensor:
        """
        Generate initial Gaussian noise in latent space using numpy.
        """
        with self.context:
            C = self.net.img_channels
            height = width = self.net.img_resolution
            shape = (C, height, width)
            noise = np.stack([np.random.RandomState(s).randn(*shape) for s in seeds], axis=0)
            return torch.from_numpy(noise).to(self.device).to(torch.float32)

    def numpy_to_pil(self, x):
        # x: (B,H,W,C) 또는 (B,C,H,W) 또는 3D (HWC/CHW), 채널 1/3/4, float[0,1] 또는 uint8
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3:
            x = x[None]  # 배치화

        # 채널-라스트로 정렬
        if x.shape[-1] not in (1, 3, 4):
            if x.shape[1] in (1, 3, 4):
                x = np.moveaxis(x, 1, -1)
            else:
                raise ValueError(f"Unsupported shape: {x.shape}")

        # uint8 변환
        if x.dtype != np.uint8:
            x = np.clip(x, 0.0, 1.0)
            x = (x * 255 + 0.5).astype(np.uint8)

        modes = {1: "L", 3: "RGB", 4: "RGBA"}
        return [Image.fromarray(img[..., 0] if img.shape[-1] == 1 else img, modes[img.shape[-1]])
                for img in x]

    def decode_vae(
        self,
        latents: torch.Tensor,
        raw_output=True,
        pil_output=False,
    ) -> Union[torch.Tensor, Image.Image]:
        """
        Decode latent tensor to image.
        """
        outputs = {}
        with self.context:
            samples = latents
            if raw_output:
                outputs['raw_output'] = samples

            if pil_output:    
                samples = (samples / 2 + 0.5).clamp(0, 1)
                samples = samples.cpu().permute(0, 2, 3, 1).float().detach().numpy()
                samples = self.numpy_to_pil(samples)
                outputs['pil_output'] = samples
            return outputs

    def get_noise_schedule(self):
        noise_schedule = NoiseScheduleEDM()
        return noise_schedule

    def get_noise(self, *, batch_size=None, seeds=None):
        assert batch_size is not None or seeds is not None
        if seeds is None:
            seeds = [42 for _ in range(batch_size)]
        
        noises = self.prepare_noise(seeds) * 80.0
        return noises

    def get_model_fn(
        self,
        noise_schedule,
        pos_conds = None,
        guidance_scale = None,
        cfg_channels = None,
    ) -> callable:

        def inner_model_fn(x, t, cond, **kwargs):
            with self.context:
                x = x.to(self.dtype)
                pred = self.net(x, t)
                return pred

        model_fn = model_wrapper(
                inner_model_fn,
                noise_schedule,
                None
        )
        return model_fn

class NoiseScheduleEDM:
    def __init__(self):
        self.total_N = 500.0 # 1 / 0.002 (sigma_min)
        self.T = 80.0 # (sigma_max)

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.zeros_like(t).to(torch.float64)

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.ones_like(t).to(torch.float64)

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t.to(torch.float64)

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """

        return -torch.log(t).to(torch.float64)

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return torch.exp(-lamb).to(torch.float64)


def model_wrapper(model, noise_schedule, class_labels=None):
    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = t_continuous
        output = model(x, t_input, cond)
        alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
        return (x - alpha_t[:, None, None, None] * output) / sigma_t[:, None, None, None]

    def model_fn(x, t_continuous):
        return noise_pred_fn(x, t_continuous, class_labels).to(torch.float64)

    return model_fn


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]        