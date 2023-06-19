from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from network import base, utils


@dataclass
class BaseTeacher(base.BaseNetwork):

    def _freeze(self):
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = False


@dataclass
class ContinuousTeacher(BaseTeacher):

    def _construct_output_layer(self):
        self._head = nn.Identity()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class ContextTeacher(BaseTeacher):

    def _construct_output_layer(self):
        self._head = nn.Identity()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x).sum(dim=1)


@dataclass
class BinaryTeacher(BaseTeacher):

    def _construct_output_layer(self):
        self._head = utils.heaviside

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class TwoHeadsTeacher(BaseTeacher):

    def _construct_output_layer(self):
        self._binary_head = utils.heaviside
        self._continuous_head = nn.Identity()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._binary_head(x), self._continuous_head(x)


@dataclass(kw_only=True)
class TwoModulesSharedTeacher(BaseTeacher):
    shared_dimension = int

    def _construct_layers(self):
        self._layers = nn.ModuleList()
        self._dimensions = [self.input_dimension] + self.hidden_dimensions
        for in_size, out_size in zip(self._dimensions[:-1],
                                     self._dimensions[1:]):
            shared = nn.Linear(in_size, self.shared_dimension, bias=self.bias)
            layer = nn.Linear(in_size, out_size, bias=self.bias)
            self._initialise_weights(layer)
            self._initialise_weights(shared)

            layer.weight.data[:, :self.shared_dimension] = shared.weight.data
            self._layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            self.layer.weight.data.mul_(self.mask_use)
            x = self._nonlinear_function(layer(x))

    def _construct_output_layer(self):
        self._module1 = nn.Identity()
        self._module2 = nn.Identity()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._module1(x), self._module2(x)