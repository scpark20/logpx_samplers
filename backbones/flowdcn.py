import torch
import numpy as np
from typing import List, Tuple, Union
from PIL import Image

from .backbone import Backbone
from solvers.common import NoiseScheduleFlow, model_wrapper

# 주의: 실제 FlowDCN 파이프라인 경로/클래스명으로 바꾸세요.
# 예: from lib.pipelines.flowdcn_pipeline import FlowDCNPipeline
from lib.pipelines.flowdcn_pipeline import FlowDCNPipeline  # <- 이 라인만 여러분 환경에 맞게 수정

class FlowDCN(Backbone):
    """
    Class-conditional Rectified-Flow backbone (FlowDCN-B/2, FlowDCN-XL/2).
    - ImageNet 1000-class 가정(CFG용 null class id=1000).
    - velocity 예측 → vector_prediction 모드에서 사용.
    """
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = 'path-or-repo-of-flowdcn',  # 로컬 경로 또는 HF repo id
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # 파이프라인 로드 (Diffusers 스타일 API 가정)
        # 모델 체크포인트가 로컬 폴더면 해당 경로를, 허깅페이스면 repo id를 넣으세요.
        self.pipe = FlowDCNPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(self.device)

        # 서브모듈 dtype & eval
        for submod in (self.pipe.vae, self.pipe.transformer):
            submod.to(dtype)
            submod.eval()

    @torch.inference_mode()
    def prepare_noise(self, seeds: List[int]) -> torch.Tensor:
        C = self.pipe.transformer.config.in_channels
        H = W = self.pipe.transformer.config.sample_size
        shape = (C, H, W)
        noise = np.stack([np.random.RandomState(s).randn(*shape) for s in seeds], axis=0)
        return torch.from_numpy(noise).to(self.device).to(torch.float32)

    @torch.inference_mode()
    def decode_vae(self, latents: torch.Tensor) -> Union[torch.Tensor, Image.Image]:
        lat = (latents / self.pipe.vae.config.scaling_factor).to(self.dtype)
        samples = self.pipe.vae.decode(lat).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        return self.pipe.numpy_to_pil(samples)

    @torch.inference_mode()
    def get_model_fn(
        self,
        pos_conds: List[int] = [0],
        guidance_scale: float = 1.375,    # 논문 기본값 근처
        seeds: Union[List[int], None] = None,
    ) -> Tuple[callable, NoiseScheduleFlow, torch.Tensor]:
        batch_size = len(pos_conds)
        if seeds is None:
            seeds = [42 for _ in range(batch_size)]
        assert len(seeds) == batch_size

        latents = self.prepare_noise(seeds)
        noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")

        class_labels = torch.tensor(pos_conds, device=self.device).reshape(-1)
        class_null = torch.tensor([1000] * len(pos_conds), device=self.device)  # 모델 정의에 맞게 조정

        @torch.inference_mode()
        def inner_model_fn(x, t, cond, **kwargs):
            x = x.to(kwargs['dtype'])
            # 파이프라인 출력 형식이 .sample일 수도 있고 텐서일 수도 있음 → 안전하게 처리
            out = self.pipe.transformer(x, timestep=t, class_labels=cond)
            try:
                pred = out.sample
            except AttributeError:
                pred = out
            # 채널 정합
            if pred.shape[1] != x.shape[1]:
                pred = pred[:, :x.shape[1]]
            return pred

        # model_type="flow"로 감싼 뒤, solver에서 algorithm_type="vector_prediction"을 쓰면
        # 최종 model_fn(x,t)는 네트워크의 velocity 출력을 그대로 돌려줍니다.
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