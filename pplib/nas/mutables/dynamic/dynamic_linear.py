import random
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from ..dynamic_mutable import DynamicMutable


class LinearSample(NamedTuple):
    sample_in_dim: int
    sample_out_dim: int


class DynamicLinear(DynamicMutable[LinearSample, LinearSample]):
    """Dynamic mutable for Linear layer.

    Args:
        max_in_dim (int): _description_
        max_out_dim (int): _description_
        bias (bool, optional): _description_. Defaults to True.
        scale (bool, optional): _description_. Defaults to False.
        alias (Optional[str], optional): _description_. Defaults to None.
        module_kwargs (Optional[Dict[str, Dict]], optional): _description_.
            Defaults to None.
        init_cfg (Optional[Dict], optional): _description_. Defaults to None.
    """

    def __init__(self,
                 max_in_dim: int,
                 max_out_dim: int,
                 bias: bool = True,
                 scale: bool = False,
                 alias: Optional[str] = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        DynamicMutable.__init__(
            self, module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

        self.max_in_dim = max_in_dim
        self.max_out_dim = max_out_dim

        # warp linear
        self.linear = nn.Linear(
            in_features=max_in_dim, out_features=max_out_dim, bias=bias)

        # store parameters
        self.samples: Dict[str, nn.Parameter] = {}
        # store args
        self._choice: LinearSample = LinearSample(max_in_dim, max_out_dim)

        # scale
        self.scale = scale

    def sample_choice(self) -> LinearSample:
        """sample with random choice"""
        return LinearSample(
            random.randint(0, self.max_in_dim),
            random.randint(0, self.max_out_dim))

    def sample_parameters(self, choice: LinearSample) -> None:
        assert choice.sample_in_dim <= self.max_in_dim, \
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
            # assert already called sample_parameters
            assert self.samples is not None, \
                'Did not call `sample_parameters`'
            return F.linear(x, self.samples['weight'],
                            self.samples['bias']) * (
                                self.samples['scale'] if self.scale else 1)

    def fix_chosen(self, chosen: LinearSample) -> None:
        """fix chosen"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLinear is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        assert chosen.sample_in_dim <= self.max_in_dim, \
            'sampled input dim is larger than max input dim.'
        assert chosen.sample_out_dim <= self.max_out_dim, \
            'sampled output dim is larger than max output dim.'

        # new a linear layer
        temp_weight = self.linear.weight.data[:chosen.sample_out_dim, :chosen.
                                              sample_in_dim]
        temp_bias = self.linear.bias.data[:chosen.sample_out_dim]
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

    def calc_sampled_flops(self, sequence_length) -> float:
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops

    def calc_sampled_params(self) -> float:
        assert 'weight' in self.samples['weight'].numel(), \
            'Please call sample_parameters before calc flops'
        weight_numel = self.samples['weight'].numel()
        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0.

        return weight_numel + bias_numel


class LinearSuper(DynamicMutable, Linear):
    """_summary_

    Args:
        super_in_dim (_type_): _description_
        super_out_dim (_type_): _description_
        bias (bool, optional): _description_. Defaults to True.
        uniform_ (_type_, optional): _description_. Defaults to None.
        non_linear (str, optional): _description_. Defaults to 'linear'.
        scale (bool, optional): _description_. Defaults to False.
    """

    def __init__(self,
                 super_in_dim: int,
                 super_out_dim: int,
                 bias: bool = True,
                 uniform_=None,
                 non_linear='linear',
                 scale: bool = False) -> None:

        Linear.__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples: Dict[str, nn.Parameter] = {}

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_out_dim, :self.
                                             sample_in_dim]
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = self.bias[:self.sample_out_dim]
        return self.samples

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

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)
