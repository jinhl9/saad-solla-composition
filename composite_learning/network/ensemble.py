from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from network import base, utils


@dataclass
class Ensemble(base.BaseEnsemble):

    def _process_input(self, x):
        return x


@dataclass(kw_only=True)
class PartitionedEnsemble(base.BaseEnsemble):
    partitions: List[int]

    def _process_input(self, x, partition_index):
        start = sum(self.partitions[:partition_index])
        end = sum(self.partitions[:partition_index + 1])
        return x[..., start:end]

    def _forward(self, network_index: int, x: torch.Tensor) -> torch.Tensor:
        x = self._process_input(x, network_index)
        return self._networks[network_index](x)
