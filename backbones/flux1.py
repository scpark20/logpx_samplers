import torch
import numpy as np
from diffusers import FluxPipeline
from typing import Optional, List, Tuple, Union, Dict, Any
from .backbone import Backbone
from solvers.common import NoiseScheduleFlow, model_wrapper
from PIL import Image


class FLUX1D(Backbone):
    """
    FLUX.1-dev backbone adapter for your (NoiseScheduleFlow, model_wrapper) interface.

    - External latent state x: (B, C_lat, H_lat, W_lat), where
        C_lat = transformer.in_channels // 4
        H_lat = 2 * (image_height // vae_scale_factor)
        W_lat = 2 * (image_width  // vae_scale_factor)

    Internally, FLUX transformer consumes packed latents:
        (B, (H_lat//2)*(W_lat//2), transformer.in_channels)
    and needs img_ids / txt_ids.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        height: int = 1024,
        width: int = 1024,
        trainable: bool = False,
    ):
        super().__init__(trainable)
        self.device = torch.device(device)
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)

        # Load and move pipeline
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast submodules and set eval
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.transformer):
            if submod is None:
                continue
            submod.to(dtype)
            submod.eval()

    def set_freeze(self):
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.transformer):
            if submod is None:
                continue
            submod.eval()
            for p in submod.parameters():
                p.requires_grad_(False)

    # ---- helpers copied/compatible with diffusers FLUX packing ----
    @staticmethod
    def _pack_latents(latents_4d: torch.Tensor) -> torch.Tensor:
        # latents_4d: (B, C, H, W) with H,W even
        b, c, h, w = latents_4d.shape
        lat = latents_4d.view(b, c, h // 2, 2, w // 2, 2)
        lat = lat.permute(0, 2, 4, 1, 3, 5)  # (B, H//2, W//2, C, 2, 2)
        lat = lat.reshape(b, (h // 2) * (w // 2), c * 4)
        return lat

    @staticmethod
    def _unpack_latents(latents_packed: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
        # height/width are *image* height/width (same as Flux pipeline convention)
        b, num_patches, ch = latents_packed.shape
        h = height // vae_scale_factor
        w = width // vae_scale_factor
        lat = latents_packed.view(b, h, w, ch // 4, 2, 2)
        lat = lat.permute(0, 3, 1, 4, 2, 5)
        lat = lat.reshape(b, ch // 4, h * 2, w * 2)
        return lat

    @staticmethod
    def _prepare_latent_image_ids(batch_size: int, h_lat: int, w_lat: int, device, dtype) -> torch.Tensor:
        # h_lat,w_lat are packed-latent spatial dims BEFORE pack() (i.e., external x has these)
        latent_image_ids = torch.zeros(h_lat // 2, w_lat // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(h_lat // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(w_lat // 2)[None, :]
        hid, wid, cid = latent_image_ids.shape
        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(batch_size, hid * wid, cid)
        return latent_image_ids.to(device=device, dtype=dtype)

    def _latent_shape_from_image(self) -> Tuple[int, int, int]:
        # external latent shape (C_lat, H_lat, W_lat)
        vae_sf = int(self.pipe.vae_scale_factor)
        c_lat = int(self.pipe.transformer.config.in_channels) // 4
        h_lat = 2 * (self.height // vae_sf)
        w_lat = 2 * (self.width // vae_sf)
        return c_lat, h_lat, w_lat

    # ---- API expected by your sampler/solver ----
    def prepare_noise(self, seeds: List[int]) -> torch.Tensor:
        with self.context:
            c_lat, h_lat, w_lat = self._latent_shape_from_image()
            shape = (c_lat, h_lat, w_lat)
            noise = np.stack([np.random.RandomState(s).randn(*shape) for s in seeds], axis=0)
            return torch.from_numpy(noise).to(self.device).to(torch.float32)

    def encode(
        self,
        pos_texts: List[str],
        neg_texts: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
          cond:   {"prompt_embeds": (B, T, D), "pooled_prompt_embeds": (B, P), "text_ids": (B, T, 3)}
          uncond: same keys
        """
        if neg_texts is None:
            neg_texts = [""] * len(pos_texts)

        # FLUX uses two prompts in diffusers API: prompt (CLIP) and prompt_2 (T5).
        # If you don't maintain separate prompt2, reuse prompt.
        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            text_ids,
        ) = self.pipe.encode_prompt(
            prompt=pos_texts,
            prompt_2=pos_texts,
            negative_prompt=neg_texts,
            negative_prompt_2=neg_texts,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
            device=self.device,
        )

        cond = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }
        uncond = {
            "prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "text_ids": text_ids,  # same shape
        }
        return cond, uncond

    def decode_vae(
        self,
        latents: torch.Tensor,
        raw_output: bool = True,
        pil_output: bool = False,
        output_type: str = "pil",
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        with self.context:
            lat = latents.to(self.dtype)

            # FLUX VAE: encode does (z - shift) * scale, so decode should invert.
            # (see diffusers Flux pipeline patterns)
            lat = (lat / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor

            img_tensor = self.pipe.vae.decode(lat, return_dict=False)[0]
            if raw_output:
                outputs["raw_output"] = img_tensor
            if pil_output:
                outputs["pil_output"] = self.pipe.image_processor.postprocess(img_tensor, output_type=output_type)
            return outputs

    def get_noise_schedule(self):
        return NoiseScheduleFlow(schedule="discrete")

    def get_noise(self, *, batch_size=None, seeds=None):
        assert batch_size is not None or seeds is not None
        if seeds is None:
            seeds = [42 for _ in range(batch_size)]
        return self.prepare_noise(seeds)

    def get_model_fn(
        self,
        noise_schedule,
        pos_conds: List[str],
        neg_conds: Optional[List[str]] = None,
        guidance_scale: float = 4.5,
    ):
        cond, uncond = self.encode(pos_conds, neg_conds)

        def inner_model_fn(x: torch.Tensor, t: torch.Tensor, cond_pack: Dict[str, torch.Tensor], **kwargs):
            """
            x: (B, C_lat, H_lat, W_lat)  (external)
            returns: same shape (B, C_lat, H_lat, W_lat)
            """
            with self.context:
                x = x.to(kwargs["dtype"])
                b, c_lat, h_lat, w_lat = x.shape

                # pack + ids
                x_packed = self._pack_latents(x)  # (B, seq, in_channels)
                img_ids = self._prepare_latent_image_ids(
                    batch_size=b,
                    h_lat=h_lat,
                    w_lat=w_lat,
                    device=x.device,
                    dtype=cond_pack["prompt_embeds"].dtype,
                )

                # timestep: keep batch shape
                if t.ndim == 0:
                    timestep = t.expand(b)
                else:
                    timestep = t
                timestep = timestep.to(dtype=x.dtype, device=x.device)

                pred_packed = self.pipe.transformer(
                    hidden_states=x_packed,
                    timestep=timestep,
                    encoder_hidden_states=cond_pack["prompt_embeds"],
                    pooled_projections=cond_pack["pooled_prompt_embeds"],
                    txt_ids=cond_pack["text_ids"],
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                # unpack back to (B, C_lat, H_lat, W_lat)
                pred = self._unpack_latents(
                    pred_packed,
                    height=self.height,
                    width=self.width,
                    vae_scale_factor=int(self.pipe.vae_scale_factor),
                )
                return pred

        model_fn = model_wrapper(
            inner_model_fn,
            noise_schedule,
            model_type="flow",
            model_kwargs={"dtype": self.dtype},
            guidance_type="classifier-free",
            condition=cond,
            unconditional_condition=uncond,
            guidance_scale=guidance_scale,
        )
        return model_fn
