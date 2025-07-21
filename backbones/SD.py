import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from typing import Tuple, Union
from .backbone import Backbone
from solvers.common import NoiseScheduleVP, model_wrapper

class SD(Backbone):
    """
    Stable-Diffusion sampler wrapping HuggingFace diffusers' StableDiffusionPipeline.
    """
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "sd-legacy/stable-diffusion-v1-5"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Load and move pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast submodules and set eval
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.unet):
            submod.to(dtype)
            submod.eval()

    @torch.inference_mode()
    def prepare_noise(
        self, batch_size: int, height: int, width: int
    ) -> torch.Tensor:
        """
        Generate initial Gaussian noise in latent space using numpy.
        """
        C = self.pipe.unet.config.in_channels
        scale = self.pipe.vae_scale_factor
        shape = (batch_size, C, height // scale, width // scale)
        noise = np.random.randn(*shape)
        return torch.from_numpy(noise).to(self.device).to(torch.float32)

    @torch.inference_mode()
    def encode(
        self, pos_text: str, neg_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize and encode positive and negative prompts for classifier-free guidance.
        """
        embeds, neg_embeds = self.pipe.encode_prompt(prompt=pos_text,
                                    device=self.device, num_images_per_prompt=1,
                                    do_classifier_free_guidance=True, negative_prompt=neg_text)
        return embeds, neg_embeds

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
        #img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
        #img = img_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
        return self.pipe.image_processor.postprocess(img_tensor, output_type=output_type)

    @torch.inference_mode()
    def get_model_fn(
        self,
        pos_text: str,
        neg_text: str = '',
        guidance_scale: float = 4.5,
        height: int = 512,
        width: int = 512,
        seed: Union[int, None] = None
    ) -> 'PIL.Image.Image':
        """
        Run a simple Euler sampling loop over num_steps timesteps.
        Seed is applied here, and noise is generated via numpy.
        """
        if seed is not None:
            np.random.seed(seed)

        embeds, neg_embeds = self.encode(pos_text, neg_text)
        latents = self.prepare_noise(1, height, width)
        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.pipe.scheduler.betas, dtype=self.dtype)

        @torch.inference_mode()
        def inner_model_fn(x, t, cond, **kwargs):
            #x = kwargs['pipe'].scheduler.scale_model_input(x, t)
            x = x.to(kwargs['dtype'])
            pred = self.pipe.unet(x, t, encoder_hidden_states=cond, return_dict=False)[0]
            return pred
        
        model_fn = model_wrapper(
                inner_model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs={"pipe": self.pipe, "dtype": self.dtype},
                guidance_type="classifier-free",
                condition=embeds,
                unconditional_condition=neg_embeds,
                guidance_scale=guidance_scale,
        )

        return model_fn, noise_schedule, latents