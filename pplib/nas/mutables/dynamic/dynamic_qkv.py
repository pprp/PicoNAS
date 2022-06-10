from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from ..dynamic_mutable import DynamicMutable


class QKVSample(NamedTuple):
    sample_in_dim: int
    sample_out_dim: int


class DynamicQKV(DynamicMutable[QKVSample, QKVSample], Linear):

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
        self._choice: QKVSample = QKVSample(max_in_dim, max_out_dim)

        # scale
        self.scale = scale

    def sample_parameters(self, choice: QKVSample) -> None:
        self._choice = choice

        sample_weight = self.weight[:, :self._choice.sample_in_dim]
        sample_weight = torch.cat([
            sample_weight[i:self._choice.sample_out_dim:3, :] for i in range(3)
        ],
                                  dim=0)  # noqa: E501

        self.samples['weight'] = sample_weight
        self.samples['bias'] = self.bias

        self.samples['scale'] = self.max_out_dim / self._choice.sample_out_dim

        if self.bias is not None:
            self.samples['bias'] = self.bias[:self._choice.sample_out_dim]

    def forward_all(self, x: Any) -> Any:
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
            # chose the lagest
            return self.forward_all(x)

    def fix_chosen(self, chosen: QKVSample) -> None:
        """fix chosen"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLinear is `fixed`. '
                'Please do not call `fix_chosen` function again.')
        # TODO
        # new a linear layer
        temp_weight = self.weight.data[:chosen.sample_out_dim, :chosen.
                                       sample_in_dim]
        temp_bias = self.bias.data[:chosen.sample_out_dim]
        self.weight = nn.Parameter(temp_weight)
        self.bias = nn.Parameter(temp_bias)

        self._choice = chosen
        self.is_fixed = True

    def forward_fixed(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def choices(self) -> List[QKVSample]:
        return super().choices

    def calc_sampled_flops(self) -> float:
        return super().calc_sampled_flops()

    def calc_sampled_params(self) -> float:
        return super().calc_sampled_params()


class qkv_super(Linear):

    def __init__(self,
                 super_in_dim,
                 super_out_dim,
                 bias=True,
                 uniform_=None,
                 non_linear='linear',
                 scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = self.sample_weight(self.weight,
                                                    self.sample_in_dim,
                                                    self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = self.sample_bias(self.bias,
                                                    self.sample_out_dim)
        return self.samples

    def sample_weight(self, weight, sample_in_dim, sample_out_dim):

        sample_weight = weight[:, :sample_in_dim]
        sample_weight = torch.cat(
            [sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim=0)

        return sample_weight

    def sample_bias(self, bias, sample_out_dim):
        sample_bias = bias[:sample_out_dim]

        return sample_bias

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (
            self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops
