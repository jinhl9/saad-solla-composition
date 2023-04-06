from dataclasses import dataclass

import torch
import torch.nn as nn

import base


@dataclass
class BaseTeacher(base.BaseNetwork):

    def _freeze(self):
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = False


@dataclass
class ContinuousTeacher(BaseTeacher):

    def _construct_output_layer(self):
        self._head = nn.Identity

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class BinaryTeacher(BaseTeacher):

    def _construct_output_layer(self):
        self._head = torch.heaviside

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class TwoHeadsTeacher(BaseTeacher):

    def _construct_output_layer(self):
        self._binary_head = torch.heaviside
        self._continuous_head = nn.Identity

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._binary_head(x), self._continuous_head(x)