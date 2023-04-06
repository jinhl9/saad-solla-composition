from dataclasses import dataclass

import torch
import torch.nn as nn

import base


@dataclass
class BaseStudent(base.BaseNetwork):
    pass


@dataclass
class ContinuousStudent(BaseStudent):

    def _construct_output_layer(self):
        self._head = nn.Identity

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class BinarysStudent(BaseStudent):

    def _construct_output_layer(self):
        self._head = torch.heaviside

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class TwoHeadsStudent(BaseStudent):

    def _construct_output_layer(self):
        self._binary_head = torch.heaviside
        self._continuous_head = nn.Identity

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._binary_head(x), self._continuous_head(x)