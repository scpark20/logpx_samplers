import torch
import numpy as np
from huggingface_hub import snapshot_download
from lib.pipelines.gmdit_pipeline import GMDiTPipeline
from lib.ops.gmflow_ops.gmflow_ops import gm_to_mean

from typing import Tuple, Union
from .backbone import Backbone
from solvers.common import NoiseScheduleFlow, model_wrapper

class GMDiT(Backbone):
    """
    DiT diffusion sampler wrapping HuggingFace diffusers' DiTPipeline.
    """
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        repo_id: str = 'Lakonik/gmflow_imagenet_k8_ema'
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Load and move pipeline
        ckpt = snapshot_download(repo_id=repo_id)
        self.pipe = GMDiTPipeline.from_pretrained(ckpt, variant='bf16', torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast submodules and set eval
        for submod in (self.pipe.vae, self.pipe.transformer):
            submod.to(dtype)
            submod.eval()

    @torch.inference_mode()
    def prepare_noise(
        self, batch_size: int,
    ) -> torch.Tensor:
        """
        Generate initial Gaussian noise in latent space using numpy.
        """
        C = self.pipe.transformer.config.in_channels
        height = width = self.pipe.transformer.config.sample_size
        shape = (batch_size, C, height, width)
        noise = np.random.randn(*shape)
        return torch.from_numpy(noise).to(self.device).to(torch.float32)

    @torch.inference_mode()
    def decode_vae(
        self,
        latents: torch.Tensor,
        output_type: str = 'pil'
    ) -> Union[torch.Tensor, 'PIL.Image.Image']:
        """
        Decode latent tensor to image.
        """
        lat = (latents / self.pipe.vae.config.scaling_factor).to(self.dtype)
        samples = self.pipe.vae.decode(lat).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = self.pipe.numpy_to_pil(samples)
        return samples

    @torch.inference_mode()
    def get_model_fn(
        self,
        class_ids = [0],
        guidance_scale: float = 4.0,
        seed: Union[int, None] = None
    ) -> 'PIL.Image.Image':
        """
        Run a simple Euler sampling loop over num_steps timesteps.
        Seed is applied here, and noise is generated via numpy.
        """
        if seed is not None:
            np.random.seed(seed)

        latents = self.prepare_noise(len(class_ids))
        noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")
        class_labels = torch.tensor(class_ids, device=self.device).reshape(-1)
        class_null = torch.tensor([1000] * len(class_ids), device=self.device)

        @torch.inference_mode()
        def inner_model_fn(x, t, cond, **kwargs):
            x = x.to(kwargs['dtype'])
            pred = self.pipe.transformer(x, timestep=t, class_labels=cond)
            return gm_to_mean(pred)
        
        model_fn = model_wrapper(
                inner_model_fn,
                noise_schedule,
                model_type="flow",
                guidance_type="classifier-free",
                model_kwargs={"dtype": self.dtype},
                condition=class_labels,
                unconditional_condition=class_null,
                guidance_scale=guidance_scale,
        )
        return model_fn, noise_schedule, latents