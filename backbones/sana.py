import torch
import numpy as np
from diffusers import SanaPipeline
from typing import Optional, List, Tuple, Union
from .backbone import Backbone
from solvers.common import NoiseScheduleFlow, model_wrapper
from PIL import Image

class SANA(Backbone):
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        model_id='Efficient-Large-Model/Sana_600M_512px_diffusers',
        #model_id='Efficient-Large-Model/Sana_600M_1024px_diffusers',
        #model_id: str = 'Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers',
        trainable = False
    ):
        super().__init__(trainable)
        self.device = torch.device(device)
        self.dtype = dtype

        # Load and move pipeline
        self.pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast submodules and set eval
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.transformer):
            submod.to(dtype)
            submod.eval()

    def set_freeze(self):
        for submod in (self.pipe.vae, self.pipe.transformer):
            submod.eval()
            for p in submod.parameters():  # 확실히 freeze
                p.requires_grad_(False)

    def prepare_noise(
        self, seeds: List[int],
    ) -> torch.Tensor:
        """
        Generate initial Gaussian noise in latent space using numpy.
        """
        with self.context:
            C = self.pipe.transformer.config.in_channels
            height = width = self.pipe.transformer.config.sample_size
            shape = (C, height, width)
            noise = np.stack([np.random.RandomState(s).randn(*shape) for s in seeds], axis=0)
            return torch.from_numpy(noise).to(self.device).to(torch.float32)

    def encode(
        self, pos_texts: List[str], neg_texts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize and encode positive and negative prompts for classifier-free guidance.
        """
        if neg_texts is None:
            neg_texts = [""] * len(pos_texts)

        embeds, attn_mask, neg_embeds, neg_mask = self.pipe.encode_prompt(
            prompt=pos_texts,
            negative_prompt=neg_texts,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            device=self.device
        )
        return embeds, attn_mask, neg_embeds, neg_mask

    def decode_vae(
        self,
        latents: torch.Tensor,
        raw_output=True,
        pil_output=False,
        output_type: str = 'pil'
    ) -> Union[torch.Tensor, Image.Image]:
        """
        Decode latent tensor to image.
        """

        outputs = {}
        with self.context:
            lat = (latents / self.pipe.vae.config.scaling_factor).to(self.dtype)
            img_tensor = self.pipe.vae.decode(lat, return_dict=False)[0]
            if raw_output:
                outputs['raw_output'] = img_tensor
            if pil_output:
                outputs['pil_output'] = self.pipe.image_processor.postprocess(img_tensor, output_type=output_type)
            return outputs

    def get_noise_schedule(self):
        noise_schedule = NoiseScheduleFlow(schedule="discrete")
        return noise_schedule

    def get_noise(self, *, batch_size=None, seeds=None):
        assert batch_size is not None or seeds is not None
        if seeds is None:
            seeds = [42 for _ in range(batch_size)]
        
        noises = self.prepare_noise(seeds)
        return noises

    def get_model_fn(
        self,
        noise_schedule,
        pos_conds: List[str],
        neg_conds: Optional[List[str]] = None,
        guidance_scale: float = 4.5,
    ) -> Tuple[callable, NoiseScheduleFlow, torch.Tensor]:
        embeds, attn_mask, neg_embeds, neg_mask = self.encode(pos_conds, neg_conds)
        
        def inner_model_fn(x, t, cond, **kwargs):
            with self.context:
                x = x.to(kwargs['dtype'])
                mask = torch.cat([kwargs['neg_mask'], kwargs['attn_mask']], dim=0)
                pred = self.pipe.transformer(x, encoder_hidden_states=cond, encoder_attention_mask=mask, timestep=t, return_dict=False)[0]
                return pred
        
        model_fn = model_wrapper(
                inner_model_fn,
                noise_schedule,
                model_type="flow",
                model_kwargs={"attn_mask": attn_mask, "neg_mask": neg_mask, "dtype": self.dtype},
                guidance_type="classifier-free",
                condition=embeds,
                unconditional_condition=neg_embeds,
                guidance_scale=guidance_scale,
        )
        return model_fn