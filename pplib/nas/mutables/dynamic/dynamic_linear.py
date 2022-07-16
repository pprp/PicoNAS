import random
from operator import mod
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from ..mutable_value import MutableValue


class DynamicLinear(Linear):
    """Dynamic mutable for Linear layer."""

    def __init__(self,
                 in_features: Union[int, MutableValue],
                 out_features: Union[int, MutableValue],
                 bias: bool = True) -> None:
        if isinstance(in_features, MutableValue):
            self.max_in_features = in_features.current_value(mode='max')
        elif isinstance(in_features, int):
            self.max_in_features = in_features
        else:
            raise f'The type of in_features {type(in_features)} is not' \
                  'supported, Only int or MutableValue is supported currently.'

        if isinstance(out_features, MutableValue):
            self.max_out_features = out_features.current_value(mode='max')
        elif isinstance(out_features, int):
            self.max_out_features = out_features
        else:
            raise f'The type of in_features {type(out_features)} is not' \
                  'supported, Only int or MutableValue is supported currently.'

        super().__init__(
            in_features=self.max_in_features,
            out_features=self.max_out_features,
            bias=bias)

        self.in_features = in_features
        self.out_features = out_features

        # store args
        self._choice = None

    def _get_weight_bias(self):
        current_in, current_out = self.max_in_features, self.max_out_features
        if isinstance(self.in_features, MutableValue):
            current_in = self.in_features

        weight = self.weight[out_mask][:, in_mask]
        bias = self.bias[out_mask] if self.bias is not None else None
        return weight, bias

    # slice
    def sample_parameters(self, choice: LinearSample) -> None:
        assert choice.sample_in_dim <= self.max_in_features, \
            'sampled input dim is larger than max input dim.'
        assert choice.sample_out_dim <= self.max_out_dim, \
            'sampled output dim is larger than max output dim.'

        self._choice = choice

        self.samples['weight'] = self.linear.weight[:self._choice.
                                                    sample_out_dim, :self.
                                                    _choice.sample_in_dim]
        self.samples['bias'] = self.linear.bias
        self.samples['scale'] = self.max_out_dim / self._choice.sample_out_dim
        if self.linear.bias is not None:
            self.samples['bias'] = self.linear.bias[:self._choice.
                                                    sample_out_dim]

    def forward_all(self, x: Any) -> Any:
        max_choice = LinearSample(self.max_in_features, self.max_out_dim)
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
            # assert already called sample_parameters
            assert self.samples is not None, \
                'Please call `sample_parameters` before forward_choice'
            return F.linear(x, self.samples['weight'],
                            self.samples['bias']) * (
                                self.samples['scale'] if self.scale else 1)

    def fix_chosen(self, chosen: LinearSample) -> None:
        """fix chosen"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLinear is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        assert chosen.sample_in_dim <= self.max_in_features, \
            'sampled input dim is larger than max input dim.'
        assert chosen.sample_out_dim <= self.max_out_dim, \
            'sampled output dim is larger than max output dim.'

        # new a linear layer
        temp_weight = self.linear.weight.data[:chosen.sample_out_dim, :chosen.
                                              sample_in_dim]
        temp_bias = self.linear.bias.data[:chosen.sample_out_dim]

        # new a linear layer
        self.linear.weight = nn.Parameter(temp_weight)
        self.linear.bias = nn.Parameter(temp_bias)

        self._choice = chosen
        self.is_fixed = True

    def forward_fixed(self, x: Tensor) -> Tensor:
        assert self.is_fixed is True, \
            'Please call fix_chosen before forward_fixed.'
        return F.linear(x, self.linear.weight, self.linear.bias)

    def choices(self) -> List[LinearSample]:
        return super().choices
