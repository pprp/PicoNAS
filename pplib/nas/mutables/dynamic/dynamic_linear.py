from typing import Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from pplib.nas.mutables.dynamic_mutable import DynamicMutable
from ..mutable_value import MutableValue


class DynamicLinear(DynamicMutable, Linear):
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
        DynamicMutable.__init__()

        self.in_features = in_features
        self.out_features = out_features

    def sample_parameters(self) -> None:
        in_features = self.get_value(self.in_features)
        out_features = self.get_value(self.out_features)

        weights = self.linear.weight[:out_features, :in_features]
        bias = self.linear.bias[:out_features]
        return weights, bias

    def forward(self, x: Tensor) -> Tensor:
        weights, bias = self.sample_parameters()
        return F.linear(x, weights, bias)

    def fix_chosen(self) -> None:
        """fix chosen and new a sliced operation"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLinear is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        in_features = self.in_features.current_value if isinstance(
            self.in_features, MutableValue) else self.in_features
        out_features = self.out_features.current_value if isinstance(
            self.out_features, MutableValue) else self.out_features

        # new a linear layer
        temp_weight = self.linear.weight.data[:out_features, :in_features]
        temp_bias = self.linear.bias.data[:out_features]

        # new a linear layer
        self.linear.weight = nn.Parameter(temp_weight)
        self.linear.bias = nn.Parameter(temp_bias)

        self.is_fixed = True

    def forward_fixed(self, x: Tensor) -> Tensor:
        assert self.is_fixed is True, \
            'Please call fix_chosen before forward_fixed.'
        return F.linear(x, self.linear.weight, self.linear.bias)

    @property
    def choices(self):
        if isinstance(self.in_features, MutableValue):
            return self.in_features.choices
