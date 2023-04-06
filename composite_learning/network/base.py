import abc
from dataclasses import dataclass, field
from typing import List
from typing import Callable
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BaseNetwork(nn.Module, abc.ABC):
    input_dimension: int = 3
    hidden_dimensions: List[int] = field(default_factory=lambda: [3])
    output_dimension: int = 1
    nonlinearity: str = "none"
    bias: bool = True
    initialisation_std: float = 1.

    def __post_init__(self):
        self._construct_layers()
        self._freeze()

    def _initialise_weights(self, layer: nn.Module, value=None) -> None:
        """In-place weight initialisation for a given layer in accordance with configuration.
        Args:
            layer: the layer to be initialised.
        """
        if value is not None:
            layer.weight.data.fill_(value)
        else:
            if self.initialisation_std is not None:
                nn.init.normal_(layer.weight, std=self.initialisation_std)
                if self.bias:
                    nn.init.normal_(layer.bias, std=self.initialisation_std)

    def _construct_layers(self):
        self._layers = []
        self._dimensions = [self.input_dimension] + self.hidden_dimensions
        for in_size, out_size in zip(self._dimensions[:-1],
                                     self._dimensions[1:]):
            layer = nn.Linear(in_size, out_size, bias=self.bias)
            self._initialise_weights(layer)
            self._layers.append(layer)

    def _get_nonlinear_function(self) -> Callable:

        if self.nonlinearity == 'none':
            return nn.Identity
        else:
            raise NotImplementedError(
                f"Undefined nonlinearity {self.nonlinearity}")

    @property
    def _nonlinear_function(self):
        return self._get_nonlinear_function()

    @abc.abstractmethod
    def _construct_output_layer(self):
        pass

    @abc.abstractmethod
    def _freeze(self):
        pass

    @property
    def layers(self):
        return self._layers

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = self._nonlinear_function(layer(x))
        self.construct_output_layer()
        y = self._get_output_from_head(x)
        return y

    @abc.abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head."""
        pass