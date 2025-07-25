import torch
import numpy as np
from diffusers import PixArtSigmaPipeline
from typing import Optional, List, Tuple, Union
from .backbone import Backbone
from solvers.common import NoiseScheduleVP, model_wrapper
from PIL import Image

class PixArtSigma(Backbone):
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Load and move pipeline
        self.pipe = PixArtSigmaPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast submodules and set eval
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.transformer):
            submod.to(dtype)
            submod.eval()

    @torch.inference_mode()
    def prepare_noise(
        self, seeds: List[int],
    ) -> torch.Tensor:
        """
        Generate initial Gaussian noise in latent space using numpy.
        """
        C = self.pipe.transformer.config.in_channels
        height = width = self.pipe.transformer.config.sample_size
        shape = (C, height, width)
        noise = np.stack([np.random.RandomState(s).randn(*shape) for s in seeds], axis=0)
        return torch.from_numpy(noise).to(self.device).to(torch.float32)

    @torch.inference_mode()
    def encode(
        self, pos_texts: List[str], neg_texts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize and encode positive and negative prompts for classifier-free guidance.
        """
        if neg_texts is None:
            neg_texts = [""] * len(pos_texts)

        embeds, attn_mask, neg_embeds, neg_mask = self.pipe.encode_prompt(prompt=pos_texts,
                                    device=self.device, num_images_per_prompt=1,
                                    do_classifier_free_guidance=True, negative_prompt=neg_texts)
        return embeds, attn_mask, neg_embeds, neg_mask

    @torch.inference_mode()
    def decode_vae(
        self,
        latents: torch.Tensor,
        output_type: str = 'pil'
    ) -> Union[torch.Tensor, Image.Image]:
        """
        Decode latent tensor to image.
        """
        
        lat = (latents / self.pipe.vae.config.scaling_factor).to(self.dtype)
        img_tensor = self.pipe.vae.decode(lat, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(img_tensor, output_type=output_type)

    @torch.inference_mode()
    def get_model_fn(
        self,
        pos_conds: List[str],
        neg_conds: Optional[List[str]] = None,
        guidance_scale: float = 4.5,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[callable, NoiseScheduleVP, torch.Tensor]:
        batch_size = len(pos_conds)
        if seeds is None:
            seeds = [42 for _ in range(batch_size)]
        assert len(seeds) == batch_size

        embeds, attn_mask, neg_embeds, neg_mask = self.encode(pos_conds, neg_conds)
        latents = self.prepare_noise(batch_size, seeds)
        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.pipe.scheduler.betas, dtype=self.dtype)

        @torch.inference_mode()
        def inner_model_fn(x, t, cond, **kwargs):
            x = x.to(kwargs['dtype'])
            mask = torch.cat([kwargs['neg_mask'], kwargs['attn_mask']], dim=0)
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            pred = self.pipe.transformer(x, timestep=t, encoder_hidden_states=cond, encoder_attention_mask=mask,
                                        added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
            pred = pred.chunk(2, dim=1)[0]
            return pred
        
        model_fn = model_wrapper(
                inner_model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs={"attn_mask": attn_mask, "neg_mask": neg_mask, "dtype": self.dtype},
                guidance_type="classifier-free",
                condition=embeds,
                unconditional_condition=neg_embeds,
                guidance_scale=guidance_scale,
        )

        return model_fn, noise_schedule, latents