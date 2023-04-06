import abc
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Base(abc.ABC):
    batch_size: int

    @abc.abstractmethod
    def get_batch(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Base data class method")
