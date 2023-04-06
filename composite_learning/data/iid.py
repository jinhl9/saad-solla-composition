from typing import Dict
from dataclasses import dataclass

import torch

import base


@dataclass
class IIDTask(base.Base):
    batch_size: int
    continuous_input_dim: int
    binary_input_dim: int
    mean: torch.FloatTensor
    variance: torch.FloatTensor

    def __post_init__(self):
        assert len(self.mean) == len(
            self.variance), "mean and variance should be same size tensors"
        assert len(self.mean) == (
            self.continuous_input_dim + self.binary_input_dim
        ), "mean and variance should match total input dimension"

        self.dist = torch.distributions.Normal(loc=self.mean,
                                               scale=torch.sqrt(self.variance))

    def get_batch(self) -> Dict[str, torch.Tensor]:

        return {'x': self.dist.sample((self.batch_size,))}
