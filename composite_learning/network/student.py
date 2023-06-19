from dataclasses import dataclass

import torch
import torch.nn as nn

from network import base, utils


@dataclass(eq=False)
class BaseStudent(base.BaseNetwork):

    def _freeze(self):
        pass


class ContinuousStudent(BaseStudent):

    def _construct_output_layer(self):
        self._head = nn.Identity()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class ContextStudent(BaseStudent):

    def _construct_output_layer(self):
        self._head = nn.Identity()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x).sum(dim=1)


class BinaryStudent(BaseStudent):

    def _construct_output_layer(self):
        self._head = nn.Sigmoid()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


@dataclass
class TwoHeadsStudent(BaseStudent):

    def _construct_output_layer(self):
        self._binary_head = utils.heaviside
        self._continuous_head = nn.Identity()

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._binary_head(x), self._continuous_head(x)