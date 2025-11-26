import torch
import numpy as np
from typing import Optional, List, Tuple, Union
from PIL import Image

from diffusers import StableDiffusionPipeline

from .backbone import Backbone
from solvers.common import NoiseScheduleVP, model_wrapper


class StableDiffusion(Backbone):
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        trainable: bool = False,
    ):
        super().__init__(trainable)
        self.device = torch.device(device)
        self.dtype = dtype

        # 파이프라인 로드 (SD / SD2.x만 가정)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.pipe.to(self.device)

        # 필요한 모듈만 캐스팅 및 eval
        for submod in (self.pipe.vae, self.pipe.unet, self.pipe.text_encoder):
            submod.to(dtype)
            submod.eval()

        # hook & activation 버퍼
        self.unet_enc_out: List[torch.Tensor] = []
        self._hook_handle = None

    # ------------------------------------------------------------------
    # bottleneck hook 유틸
    # ------------------------------------------------------------------
    def register_hook(self):
        """
        U-Net mid_block(bottleneck)에 forward hook 등록.
        이후 UNet가 한 번 호출될 때마다
        self.unet_enc_out 에 bottleneck activation을 append.
        """
        self.unet_enc_out = []

        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            self.unet_enc_out.append(out.detach())

        # 기존 hook 있으면 제거 후 다시 등록
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        self._hook_handle = self.pipe.unet.mid_block.register_forward_hook(hook_fn)

    def clear_hook(self):
        """
        hook 제거 + 버퍼 비우기.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.unet_enc_out = []

    def set_freeze(self):
        # 학습 안 할 모듈 확실히 freeze
        for submod in (self.pipe.vae, self.pipe.unet, self.pipe.text_encoder):
            submod.eval()
            for p in submod.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Latent noise 생성
    # ------------------------------------------------------------------
    def prepare_noise(self, seeds: List[int]) -> torch.Tensor:
        """
        Stable Diffusion latent 공간 초기 노이즈 생성.
        shape = (B, C, H, W)
        """
        with self.context:
            C = self.pipe.unet.config.in_channels
            height = width = self.pipe.unet.config.sample_size
            shape = (C, height, width)
            noise = np.stack(
                [np.random.RandomState(s).randn(*shape) for s in seeds],
                axis=0,
            )
            return torch.from_numpy(noise).to(self.device).to(torch.float32)

    # ------------------------------------------------------------------
    # 텍스트 인코딩 (SD/SD2.x만)
    # ------------------------------------------------------------------
    def encode(
        self,
        pos_texts: List[str],
        neg_texts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CFG용 positive/negative 프롬프트 인코딩.
        SD/SD2.x: encode_prompt → (prompt_embeds, negative_prompt_embeds)만 반환한다고 가정.
        """
        if neg_texts is None:
            neg_texts = [""] * len(pos_texts)

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=pos_texts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=neg_texts,
        )

        embeds = prompt_embeds
        neg_embeds = negative_prompt_embeds

        # attention mask는 내부에서 안 쓰더라도, 인터페이스 유지용으로 ones 생성
        attn_mask = torch.ones(
            embeds.shape[:2],  # (B, L)
            dtype=torch.bool,
            device=self.device,
        )
        neg_mask = torch.ones(
            neg_embeds.shape[:2],
            dtype=torch.bool,
            device=self.device,
        )

        return embeds, attn_mask, neg_embeds, neg_mask

    # ------------------------------------------------------------------
    # VAE 디코딩
    # ------------------------------------------------------------------
    def decode_vae(
        self,
        latents: torch.Tensor,
        raw_output: bool = True,
        pil_output: bool = False,
        output_type: str = "pil",
    ) -> Union[torch.Tensor, Image.Image, dict]:
        """
        Latent → 이미지 복원.
        PixArtAlpha 버전과 동일한 인터페이스(outputs dict).
        """
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

    # ------------------------------------------------------------------
    # 노이즈 스케줄 및 노이즈 샘플링
    # ------------------------------------------------------------------
    def get_noise_schedule(self):
        noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            betas=self.pipe.scheduler.betas,
            dtype=self.dtype,
        )
        return noise_schedule

    def get_noise(self, *, batch_size=None, seeds=None):
        assert batch_size is not None or seeds is not None
        if seeds is None:
            seeds = [42 for _ in range(batch_size)]
        noises = self.prepare_noise(seeds)
        return noises

    # ------------------------------------------------------------------
    # Solver에서 호출할 model_fn
    # ------------------------------------------------------------------
    def get_model_fn(
        self,
        noise_schedule,
        pos_conds: List[str],
        neg_conds: Optional[List[str]] = None,
        guidance_scale: float = 7.5,
    ):
        """
        외부 solver에서 사용할 model_fn 생성.
        - model_type="noise": UNet이 예측하는 것은 ε.
        - guidance_type="classifier-free": CFG는 model_wrapper가 처리.
        """
        embeds, attn_mask, neg_embeds, neg_mask = self.encode(pos_conds, neg_conds)

        def inner_model_fn(x, t, cond, **kwargs):
            with self.context:
                x = x.to(kwargs["dtype"])
                eps = self.pipe.unet(
                    x,
                    t,
                    encoder_hidden_states=cond,
                    return_dict=False,
                )[0]
                return eps

        model_fn = model_wrapper(
            inner_model_fn,
            noise_schedule,
            model_type="noise",
            model_kwargs={
                "attn_mask": attn_mask,  # 인터페이스 맞춤용, 실제로는 안 씀
                "neg_mask": neg_mask,
                "dtype": self.dtype,
            },
            guidance_type="classifier-free",
            condition=embeds,
            unconditional_condition=neg_embeds,
            guidance_scale=guidance_scale,
        )

        return model_fn
