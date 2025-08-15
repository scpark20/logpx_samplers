from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
from PIL import Image

class Backbone(ABC):
    """
    Base class for diffusion pipelines.
    Subclasses must implement get_model_fn and decode_vae methods.
    """
    def __init__(self):
        super().__init__()
        self.pipe = None

    @abstractmethod
    def get_model_fn(self, *args, **kwargs) -> Tuple[callable, object, torch.Tensor]:
        """
        Prepares and returns the model function for the solver, 
        the noise schedule, and initial latents.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_vae(self, latents: torch.Tensor) -> Union[torch.Tensor, Image.Image]:
        """
        Decode latent tensor to image.
        """
        raise NotImplementedError