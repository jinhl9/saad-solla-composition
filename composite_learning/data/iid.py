from typing import Dict
from typing import List
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


@dataclass
class IIDRLTask(base.BaseDataLoader):
    """IID task data modules.

    Datapoints are sampled from unit nomral gaussian in batch, size of [batch_size, continuous_input_dim + binary_input_dim]

    Args:
        seq_len: length of input sequence.
        batch_size: batch size. 
        input_dim: size of dimension where continuous value is used. 
        
    """
    seq_len: int
    batch_size: int
    input_dim: int

    def __post_init__(self):

        self.mean = torch.tensor([0.])
        self.variance = torch.tensor([1.])
        self.dist = torch.distributions.Normal(loc=self.mean,
                                               scale=torch.sqrt(self.variance))

    def get_batch(self) -> Dict[str, torch.Tensor]:
        return {
            'x':
                self.dist.sample((
                    self.batch_size,
                    self.seq_len,
                    self.input_dim,
                )).squeeze()
        }


@dataclass
class TransientRLTask(base.BaseDataLoader):
    """Transient task data modules.

    Datapoints are sampled from unit nomral gaussian in batch, size of [batch_size, input_dim]

    Args:
        seq_len_list: a list of length of input sequence from different gaussian distribution.
        batch_size: batch size. 
        input_dim: size of dimension where continuous value is used. 
        
    """
    seq_len: List[int]
    batch_size: int
    input_dim: int
    identical: bool

    def __post_init__(self):

        self.mean = torch.tensor([0.])
        self.variance = torch.tensor([1.])
        self.dist = torch.distributions.Normal(loc=self.mean,
                                               scale=torch.sqrt(self.variance))

    def get_batch(self) -> Dict[str, torch.Tensor]:
        if not self.identical:
            return {
                'x': [
                    self.dist.sample(
                        (self.batch_size, l, self.input_dim)).squeeze(dim=-1)
                    for _, l in enumerate(self.seq_len)
                ]
            }
        else:
            x = self.dist.sample(
                        (self.batch_size, self.seq_len[0], self.input_dim)).squeeze(dim=-1)
            return {
                'x': [ x for _ in self.seq_len]}
                    
    def __str__(self):
        return f"TransientRLTask(seq_len:{self.seq_len}, batch_size:{self.batch_size}, input_dim:{self.input_dim})"
