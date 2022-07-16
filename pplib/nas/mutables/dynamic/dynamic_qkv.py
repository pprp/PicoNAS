from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from pplib.nas.mutables.base_mutable import CHOICE_TYPE
from ..dynamic_mutable import DynamicMutable


class QKVSample(NamedTuple):
    sample_in_dim: int
    sample_out_dim: int


class DynamicQKV(DynamicMutable[QKVSample, QKVSample]):
    """Dynamic QKV in transformer.

    Args:
        max_in_dim (int): _description_
        max_out_dim (int): _description_
        bias (bool, optional): _description_. Defaults to True.
        scale (bool, optional): _description_. Defaults to False.
        alias (Optional[str], optional): _description_. Defaults to None.
        module_kwargs (Optional[Dict[str, Dict]], optional): _description_.
            Defaults to None.
        init_cfg (Optional[Dict], optional): _description_.
            Defaults to None.
    """

    def __init__(self,
                 max_in_dim: int,
                 max_out_dim: int,
                 bias: bool = True,
                 scale: bool = False,
                 alias: Optional[str] = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

        self.max_in_dim = max_in_dim
        self.max_out_dim = max_out_dim

        self.linear = nn.Linear(max_in_dim, max_out_dim, bias)

        # store parameters
        self.samples: Dict = {}
        # store args
        self._choice: QKVSample = QKVSample(max_in_dim, max_out_dim)

        # scale
        self.scale = scale

    def sample_choice(self) -> CHOICE_TYPE:
        return super().sample_choice()

    def sample_parameters(self, choice: QKVSample) -> None:
        self._choice = choice

        sample_weight = self.linear.weight[:, :self._choice.sample_in_dim]
        sample_weight = torch.cat([
            sample_weight[i:self._choice.sample_out_dim:3, :] for i in range(3)
        ],
                                  dim=0)  # noqa: E501

        self.samples['weight'] = sample_weight
        self.samples['bias'] = self.linear.bias

        self.samples['scale'] = self.max_out_dim / self._choice.sample_out_dim

        if self.linear.bias is not None:
            self.samples['bias'] = self.linear.bias[:self._choice.
                                                    sample_out_dim]

    def forward_all(self, x: Tensor) -> Tensor:
        max_choice = QKVSample(self.max_in_dim, self.max_out_dim)
        self.sample_parameters(max_choice)
        return F.linear(x, self.samples['weight'], self.samples['bias'])

    def forward_choice(self,
                       x: Tensor,
                       choice: Optional[QKVSample] = None) -> Tensor:
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

    def fix_chosen(self, chosen: QKVSample) -> None:
        """fix chosen"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLinear is `fixed`. '
                'Please do not call `fix_chosen` function again.')
        # TODO
        # new a linear layer
        temp_weight = self.linear.weight.data[:chosen.sample_out_dim, :chosen.
                                              sample_in_dim]
        temp_bias = self.linear.bias.data[:chosen.sample_out_dim]
        self.linear.weight = nn.Parameter(temp_weight)
        self.linear.bias = nn.Parameter(temp_bias)

        self._choice = chosen
        self.is_fixed = True

    def forward_fixed(self, x: Tensor) -> Tensor:
        return F.linear(x, self.linear.weight, self.linear.bias)

    def choices(self) -> List[QKVSample]:
        return super().choices

    def calc_sampled_flops(self, x: Any) -> float:
        total_flops = 0
        total_flops += x * np.prod(self.samples['weight'].size())
        return total_flops
