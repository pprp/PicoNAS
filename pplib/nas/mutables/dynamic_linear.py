from typing import Any, Dict, NamedTuple, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from .dynamic_mutable import DynamicMutable


class LinearSample(NamedTuple):
    sample_in_dim: int
    sample_out_dim: int


class DynamicLinear(DynamicMutable[LinearSample, LinearSample], Linear):

    def __init__(self,
                 max_in_dim: int,
                 max_out_dim: int,
                 bias: bool = True,
                 scale: bool = False,
                 alias: Optional[str] = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        DynamicMutable.__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)
        Linear.__init__(
            in_features=max_in_dim, out_features=max_out_dim, bias=bias)

        self.max_in_dim = max_in_dim
        self.max_out_dim = max_out_dim

        # store parameters
        self.samples: Dict = {}
        # store args
        self._choice: LinearSample = LinearSample(max_in_dim, max_out_dim)

        # scale
        self.scale = scale

    def sample_parameters(self, choice: LinearSample) -> None:
        self._choice = choice

        self.samples['weight'] = self.weight[:self._choice.sample_out_dim, :
                                             self._choice.sample_in_dim]
        self.samples['bias'] = self.bias

        self.samples['scale'] = self.max_out_dim / self._choice.sample_out_dim

        if self.bias is not None:
            self.samples['bias'] = self.bias[:self._choice.sample_out_dim]

    def forward_all(self, x: Any) -> Any:
        max_choice = LinearSample(self.max_in_dim, self.max_out_dim)
        self.sample_parameters(max_choice)
        return F.linear(x, self.samples['weight'], self.samples['bias'])

    def forward_choice(self,
                       x: Tensor,
                       choice: Optional[LinearSample] = None) -> Tensor:
        if choice is not None:
            self.sample_parameters(choice)
            return F.linear(x, self.samples['weight'],
                            self.samples['bias']) * (
                                self.samples['scale'] if self.scale else 1)
        else:
            # chose the lagest
            return self.forward_all(x)

    def fix_chosen(self, chosen: LinearSample) -> None:
        """fix chosen"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLinear is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        # new a linear layer
        temp_weight = self.weight.data[:chosen.sample_out_dim, :chosen.
                                       sample_in_dim]
        temp_bias = self.bias.data[:chosen.sample_out_dim]
        self.weight = nn.Parameter(temp_weight)
        self.bias = nn.Parameter(temp_bias)

        self._chosen = chosen
        self.is_fixed = True

    def forward_fixed(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
