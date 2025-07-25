import torch
import numpy as np
from typing import Tuple, Union

class Backbone:
    """
    Base class for diffusion pipelines. Subclasses must implement encode and sample methods.
    """
    def __init__(self):
        self.pipe = None

    def encode(
        self,
        positive_text: str,
        negative_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode prompts into embeddings and attention masks.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs) -> torch.Tensor:
        """
        Run the sampling procedure to generate an image tensor.
        """
        raise NotImplementedError

