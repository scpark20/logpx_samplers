import torch
import numpy as np
# from diffusers import SanaPipeline
from diffusers import AuraFlowPipeline
from typing import Tuple, Union
from .backbone import Backbone
from solvers.common import NoiseScheduleFlow, model_wrapper

class AuraFlow(Backbone):
    """
    AuraFlow diffusion sampler wrapping HuggingFace diffusers' AuraFlowPipeline.
    """
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "fal/AuraFlow"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Load and move pipeline
        self.pipe = AuraFlowPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast submodules and set eval
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.transformer):
            submod.to(dtype)
            submod.eval()

    @torch.inference_mode()
    def prepare_noise(
        self, batch_size: int, height: int, width: int
    ) -> torch.Tensor:
        """
        Generate initial Gaussian noise in latent space using numpy.
        """
        C = self.pipe.transformer.config.in_channels
        scale = self.pipe.vae_scale_factor
        shape = (batch_size, C, height // scale, width // scale)
        noise = np.random.randn(*shape)
        return torch.from_numpy(noise).to(self.device).to(self.dtype)

    @torch.inference_mode()
    def encode(
        self, pos_text: str, neg_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize and encode positive and negative prompts for classifier-free guidance.
        """
        embeds, attn_mask, neg_embeds, neg_mask = self.pipe.encode_prompt(
            prompt=pos_text,
            negative_prompt=neg_text,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            device=self.device
        )
        return embeds, attn_mask, neg_embeds, neg_mask

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

        img_tensor = self.pipe.vae.decode(lat, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(img_tensor, output_type=output_type)

    @torch.inference_mode()
    def get_model_fn(
        self,
        pos_text: str,
        neg_text: str = '',
        guidance_scale: float = 4.5,
        num_steps: int = 10,
        height: int = 1024,
        width: int = 1024,
        seed: Union[int, None] = None
    ) -> 'PIL.Image.Image':
        """
        Run a simple Euler sampling loop over num_steps timesteps.
        Seed is applied here, and noise is generated via numpy.
        """
        if seed is not None:
            np.random.seed(seed)

        embeds, attn_mask, neg_embeds, neg_mask = self.encode(pos_text, neg_text)
        latents = self.prepare_noise(1, height, width)

        noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")

        @torch.inference_mode()
        def inner_model_fn(x, t, cond, **kwargs):
            # aura use timestep value between 0 and 1, with t=1 as noise and t=0 as the image
            pred = self.pipe.transformer(x, encoder_hidden_states=cond, timestep=t/1000, return_dict=False)[0]
            return pred
        
        model_fn = model_wrapper(
                inner_model_fn,
                noise_schedule,
                model_type="flow",
                model_kwargs={"attn_mask": attn_mask, "neg_mask": neg_mask},
                guidance_type="classifier-free",
                condition=embeds,
                unconditional_condition=neg_embeds,
                guidance_scale=guidance_scale,
        )
        return model_fn, noise_schedule, latents