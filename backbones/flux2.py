import torch
import numpy as np
from diffusers import Flux2Pipeline
from typing import Optional, List, Tuple, Union
from .backbone import Backbone
from solvers.common import NoiseScheduleFlow, model_wrapper
from PIL import Image


class FLUX2(Backbone):
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id="black-forest-labs/FLUX.2-dev",
        trainable=False,
    ):
        super().__init__(trainable)
        self.device = torch.device(device)
        self.dtype = dtype

        # Load pipeline (initially on CPU)
        self.pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast and freeze main modules
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.transformer):
            submod.to(dtype)
            submod.eval()

    def set_freeze(self):
        for submod in (self.pipe.vae, self.pipe.transformer):
            submod.eval()
            for p in submod.parameters():
                p.requires_grad_(False)

    # ----------------------------------------------------------------------
    # Noise preparation
    # ----------------------------------------------------------------------
    def prepare_noise(self, seeds: List[int]) -> torch.Tensor:
        """
        Generate initial Gaussian noise in latent space.
        """
        with self.context:
            C = self.pipe.vae.config.latent_channels
            H = W = 128
            shape = (C, H, W)

            noise = np.stack(
                [np.random.RandomState(s).randn(*shape) for s in seeds],
                axis=0,
            )
            return torch.from_numpy(noise).to(self.device).to(torch.float32)

    # ----------------------------------------------------------------------
    # Prompt encoding (CFG)
    # ----------------------------------------------------------------------
    def encode(
        self,
        pos_texts: List[str],
        neg_texts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if neg_texts is None:
            neg_texts = [""] * len(pos_texts)

        embeds, txt_ids = self.pipe.encode_prompt(
            prompt=pos_texts,
            num_images_per_prompt=1,
            device=self.device,
        )

        neg_embeds, neg_txt_ids = self.pipe.encode_prompt(
            prompt=neg_texts,
            num_images_per_prompt=1,
            device=self.device,
        )
        return embeds, txt_ids, neg_embeds, neg_txt_ids

    # ----------------------------------------------------------------------
    # VAE decoding
    # ----------------------------------------------------------------------
    def decode_vae(
        self,
        latents: torch.Tensor,
        raw_output=True,
        pil_output=False,
        output_type: str = "pil",
    ):
        outputs = {}
        with self.context:
            lat = (latents / self.pipe.vae.config.scaling_factor).to(self.dtype)

            img_tensor = self.pipe.vae.decode(lat, return_dict=False)[0]

            if raw_output:
                outputs["raw_output"] = img_tensor

            if pil_output:
                outputs["pil_output"] = self.pipe.image_processor.postprocess(
                    img_tensor,
                    output_type=output_type,
                )

            return outputs

    # ----------------------------------------------------------------------
    # Noise schedule
    # ----------------------------------------------------------------------
    def get_noise_schedule(self):
        return NoiseScheduleFlow(schedule="discrete")

    # ----------------------------------------------------------------------
    # Noise generation (public entry)
    # ----------------------------------------------------------------------
    def get_noise(self, *, batch_size=None, seeds=None):
        assert batch_size is not None or seeds is not None

        if seeds is None:
            seeds = [42 for _ in range(batch_size)]

        return self.prepare_noise(seeds)

    # ----------------------------------------------------------------------
    # Model wrapper (CFG + solver-compatible)
    # ----------------------------------------------------------------------
    def get_model_fn(
        self,
        noise_schedule,
        pos_conds: List[str],
        neg_conds: Optional[List[str]] = None,
        guidance_scale: float = 4.5,
    ):
        embeds, txt_ids, neg_embeds, neg_txt_ids = self.encode(pos_conds, neg_conds)

        def inner_model_fn(x, t, cond, **kwargs):
            with self.context:
                x = x.to(kwargs["dtype"])

                # FLUX main transformer call
                pred = self.pipe.transformer(
                    x,
                    encoder_hidden_states=cond,
                    txt_ids=kwargs['txt_ids'],
                    timestep=t,
                    return_dict=False,
                )[0]

                return pred

        model_fn = model_wrapper(
            inner_model_fn,
            noise_schedule,
            model_type="flow",
            model_kwargs={
                "txt_ids": txt_ids,
                "dtype": self.dtype,
            },
            guidance_type="classifier-free",
            condition=embeds,
            unconditional_condition=neg_embeds,
            guidance_scale=guidance_scale,
        )

        return model_fn
