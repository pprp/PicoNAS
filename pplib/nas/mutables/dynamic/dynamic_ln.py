from typing import Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm

from pplib.nas.mutables.dynamic_mixin import DynamicMixin
from ..mutable_value import MutableValue


class DynamicLayerNorm(DynamicMixin, LayerNorm):
    """Dynamic mutable for Linear layer."""

    def __init__(
        self,
        normalized_shape: Union[int, MutableValue],
        eps: float = 1e-5,
        bias: bool = True,
        mode: str = 'max',
    ) -> None:

        if isinstance(normalized_shape, MutableValue):
            # TODO get max
            self.max_normalized_shape = normalized_shape.current_value
        elif isinstance(normalized_shape, int):
            self.max_normalized_shape = normalized_shape
        else:
            raise f'The type of normalized_shape {type(normalized_shape)} is ' 'not supported, Only int or MutableValue is supported.'

        super(DynamicLayerNorm, self).__init__(
            normalized_shape=self.max_normalized_shape, eps=eps)

        self.normalized_shape = normalized_shape

        # flags required by dynamic mixin
        self._mode = mode
        self._is_fixed = False

    def sample_parameters(self) -> None:
        normalized_shape = self.get_value(self.normalized_shape)

        weights = self.weight[:normalized_shape]
        bias = self.bias[:normalized_shape] if self.bias is not None else None
        return normalized_shape, weights, bias

    def forward(self, x: Tensor) -> Tensor:
        normalized_shape, weights, bias = self.sample_parameters()
        return F.layer_norm(x, (normalized_shape, ), weights, bias, self.eps)

    def fix_chosen(self) -> None:
        """fix chosen and new a sliced operation"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLayerNorm is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        normalized_shape = (
            self.normalized_shape.current_value if isinstance(
                self.normalized_shape, MutableValue) else
            self.normalized_shape)

        # new a layer
        temp_weight = self.weight.data[:normalized_shape]
        temp_bias = self.bias.data[:normalized_shape]

        # new a layer
        self.weight = nn.Parameter(temp_weight)
        self.bias = nn.Parameter(temp_bias)

        self.is_fixed = True

    def forward_fixed(self, x: Tensor) -> Tensor:
        assert self.is_fixed is True, 'Please call fix_chosen before forward_fixed.'
        normalized_shape = self.get_value(self.normalized_shape)
        return F.layer_norm(x, (normalized_shape, ), self.weight, self.bias,
                            self.eps)

    @property
    def choices(self):
        if isinstance(self.normalized_shape, MutableValue):
            return self.normalized_shape.choices
