import abc
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class BaseDataLoader(abc.ABC):
    """
    Abstract base class for data modules. 
    """
    batch_size: int

    @abc.abstractmethod
    def get_batch(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Base data class method")