import torch
import numpy as np
from diffusers import FluxPipeline
from typing import Tuple, Union
from .backbone import Backbone
from solvers.common import NoiseScheduleFlow, model_wrapper

class Flux(Backbone):
    """
    Flux diffusion sampler wrapping HuggingFace diffusers' FluxPipeline.
    """
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "black-forest-labs/FLUX.1-dev"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Load and move pipeline
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # Cast submodules and set eval
        for submod in (self.pipe.vae, self.pipe.text_encoder, self.pipe.transformer):
            submod.to(dtype)
            submod.eval()

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):

        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @torch.inference_mode()
    def prepare_noise(
        self, batch_size: int, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate initial Gaussian noise in latent space using numpy.
        """
        C = self.pipe.transformer.config.in_channels // 4
        scale = self.pipe.vae_scale_factor
        height = 2*(int(height) // (scale*2))
        width = 2*(int(width) // (scale*2))
        shape = (batch_size, C, height, width)
        # noise = np.random.randn(*shape)
        # noise = torch.from_numpy(noise).to(self.device).to(self.dtype)
        noise = torch.randn(shape)
        
        noise = self._pack_latents(noise, batch_size, C, height, width)
        
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, self.device, self.dtype)
        # return torch.from_numpy(noise).to(self.device).to(self.dtype), latent_image_ids, noise
        return noise.to(self.device).to(self.dtype), latent_image_ids

    @torch.inference_mode()
    def encode(
        self, pos_text: str, neg_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize and encode positive and negative prompts for classifier-free guidance.
        """
        # embeds, attn_mask, neg_embeds, neg_mask = self.pipe.encode_prompt(
        embeds, pooled_embeds, text_ids = self.pipe.encode_prompt(
            prompt=pos_text,
            prompt_2=None,
            # negative_prompt=neg_text,
            num_images_per_prompt=1,
            # do_classifier_free_guidance=True,
            device=self.device
        )
        
        # TODO
        do_classifier_free_guidance=True
        neg_embeds = None
        neg_pooled_embeds = None
        neg_text_ids = None
        if do_classifier_free_guidance:
            neg_embeds, neg_pooled_embeds, neg_text_ids = self.pipe.encode_prompt(
                prompt=neg_text,
                prompt_2=None,
                num_images_per_prompt=1,
                # do_classifier_free_guidance=True,
                device=self.device
            )
        
        return embeds, pooled_embeds, text_ids, neg_embeds, neg_pooled_embeds, neg_text_ids

    @torch.inference_mode()
    def decode_vae(
        self,
        latents: torch.Tensor,
        output_type: str = 'pil'
    ) -> Union[torch.Tensor, 'PIL.Image.Image']:
        """
        Decode latent tensor to image.
        """
        
        # TODO height, width 추가 변수 deocde 시 도는 init 함수 내부에 작성 필요 (_unpack_latents)
        latents = self._unpack_latents(latents, 1024, 1024, self.pipe.vae_scale_factor)
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

        # embeds, attn_mask, neg_embeds, neg_mask = self.encode(pos_text, neg_text)
        embeds, pooled_embeds, text_ids, neg_embeds, neg_pooled_embeds, neg_text_ids = self.encode(pos_text, neg_text)
        latents, latent_image_ids = self.prepare_noise(1, height, width)

        noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")
        # TODO true_cfg_scale 변수 수정 필요
        true_cfg_scale = 1.0
        do_classifier_free_guidance=true_cfg_scale > 1 and neg_text is not None
        joint_attention_kwargs = {}

        @torch.inference_mode()
        def inner_model_fn(x, t, cond, **kwargs):
            # flux use timestep value between 0 and 1, with t=1 as noise and t=0 as the image
            # flux에서는 positive, negative prompt 따로 진행
            pred = self.pipe.transformer(
                hidden_states=x,
                timestep=t / 1000,
                guidance=torch.full([1], guidance_scale, device=self.device, dtype=torch.float32),
                pooled_projections=pooled_embeds,
                encoder_hidden_states=embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                neg_pred = self.pipe.transformer(
                    hidden_states=x,
                    timestep=t / 1000,
                    guidance=torch.full([1], guidance_scale, device=self.device, dtype=torch.float32),
                    pooled_projections=neg_pooled_embeds,
                    encoder_hidden_states=neg_embeds,
                    txt_ids=neg_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
                pred = neg_pred + true_cfg_scale * (pred - neg_pred)

            return pred
        
        model_fn = model_wrapper(
                inner_model_fn,
                noise_schedule,
                model_type="flow",
                # model_kwargs={"attn_mask": attn_mask, "neg_mask": neg_mask},
                guidance_type="classifier-free",
                condition=embeds,
                # unconditional_condition=neg_embeds,
                guidance_scale=guidance_scale,
        )
        return model_fn, noise_schedule, latents