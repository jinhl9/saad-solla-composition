from typing import Dict
from dataclasses import dataclass

import torch

from data import base


@dataclass
class IIDTask(base.BaseDataLoader):
    """IID task data modules.

    Datapoints are sampled from unit nomral gaussian in batch, size of [batch_size, continuous_input_dim + binary_input_dim]

    Args:
        batch_size: batch size. 
        continuous_input_dim: size of dimension where continuous value is used. 
        binary_input_dim: size of dimension where binary value is used. 

    """
    batch_size: int
    continuous_input_dim: int
    binary_input_dim: int
    partition_num: int = 1

    def __post_init__(self):
        assert self.continuous_input_dim + self.binary_input_dim > 0, "dimension should be at least 1"
        assert (
            self.continuous_input_dim + self.binary_input_dim
        ) % self.partition_num == 0, "Total number of features should be divisible by the partition numbers"
        self.partition_size = int(
            (self.continuous_input_dim + self.binary_input_dim) //
            self.partition_num)
        self.mean = torch.tensor([0.])
        self.variance = torch.tensor([1.])
        self.dist = torch.distributions.Normal(loc=self.mean,
                                               scale=torch.sqrt(self.variance))

    def get_batch(self) -> Dict[str, torch.Tensor]:
        if self.partition_num == 1:
            return {
                'x':
                    self.dist.sample((
                        self.batch_size,
                        self.continuous_input_dim + self.binary_input_dim,
                    )).squeeze()
            }
        else:
            return {
                'x':
                    self.dist.sample((
                        self.batch_size,
                        self.partition_num,
                        self.partition_size,
                    )).squeeze()
            }
