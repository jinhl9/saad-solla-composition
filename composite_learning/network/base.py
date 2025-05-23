import abc
from dataclasses import dataclass, field
from typing import List
from typing import Callable
from typing import Union
from typing import Optional
import math
import torch
import torch.nn as nn
from network import utils


@dataclass(eq=False)
class BaseNetwork(nn.Module, abc.ABC):
    input_dimension: int = 3
    hidden_dimensions: List[int] = field(default_factory=lambda: [3])
    output_dimension: int = 1
    nonlinearity: str = "none"
    normalize: bool = False
    standardize: bool = False
    weights: Optional[List[torch.FloatTensor]] = None
    bias: bool = False
    initialisation_std: float = 1.
    name: str = 'Net'

    def __post_init__(self):
        super(BaseNetwork, self).__init__()
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
                if self.standardize:
                    layer.weight.data /= math.sqrt(
                        (layer.weight.data @ layer.weight.data.T /
                         self.input_dimension).item())

    def _construct_layers(self):
        self._layers = nn.ModuleList()
        self._dimensions = [self.input_dimension] + self.hidden_dimensions
        for i, (in_size, out_size) in enumerate(
                zip(
                    self._dimensions[:-1],
                    self._dimensions[1:],
                )):
            layer = nn.Linear(in_size, out_size, bias=self.bias)
            if self.weights is None:
                layer_value = None
            else:
                layer_value = self.weights[i]
            self._initialise_weights(layer, value=layer_value)
            self._layers.append(layer)

    def _get_nonlinear_function(self) -> Callable:

        if self.nonlinearity == 'none':
            return nn.Identity()
        elif self.nonlinearity == 'scaled_sigmoid':
            return utils.scaled_sigmoid
        elif self.nonlinearity == 'sign':
            return torch.sign
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
            _y = layer(x)
            if self.normalize:
                _y = _y / math.sqrt(self.input_dimension)
            x = self._nonlinear_function(_y)
        self._construct_output_layer()
        y = self._get_output_from_head(x)
        return _y, y

    @abc.abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head."""
        pass


@dataclass(eq=False)
class BaseEnsemble(abc.ABC):
    networks: List[BaseNetwork]

    @abc.abstractmethod
    def _process_input(self, x, **kwargs):
        return NotImplementedError

    @property
    def _networks(self):
        return self.networks

    def parameters(self):
        params_list = []
        for net in self.networks:
            params_list += list(net.parameters())
        return params_list

    @property
    def names(self):
        return [network.name for network in self.networks]

    @property
    def num_networks(self):
        return len(self._networks)

    def _forward(self, network_index: int, x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        x = self._process_input(x, **kwargs)
        return self._networks[network_index](x)

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return [self._forward(i, x, **kwargs) for i in range(self.num_networks)]


@dataclass(eq=False)
class SharedEnsembleBase(abc.ABC):
    networks: List[Union[List[BaseNetwork], BaseNetwork]]
    shared_dimensions: List[int]

    def __post_init__(self):
        for network_index, shared_d in enumerate(self.shared_dimensions):
            if shared_d != 0:
                _hidden_size = self.networks[network_index][
                    0].hidden_dimensions[0]
                self._shared_weights = nn.Linear(shared_d, _hidden_size)
                for net in self.networks[network_index]:
                    net.layers[
                        0].weight.data[:, :
                                       shared_d] = self._shared_weights.weight.data
        self.networks = [
            net for net_group in self.networks for net in net_group
        ]

    @property
    def _networks(self):
        return self.networks

    def parameters(self):
        params_list = []
        for net in self.networks:
            params_list += list(net.parameters())
        return params_list

    @property
    def names(self):
        return [network.name for network in self.networks]

    @property
    def num_networks(self):
        return len(self._networks)

    def _forward(self, network_index: int, x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        x = self._process_input(x, **kwargs)
        return self._networks[network_index](x)

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return [self._forward(i, x, **kwargs) for i in range(self.num_networks)]
