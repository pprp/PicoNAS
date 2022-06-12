import random
from typing import Any, Dict, List, NamedTuple, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from ..dynamic_mutable import DynamicMutable


class LayerNormSample(NamedTuple):
    embed_dim: int


class DynamicLayerNorm(DynamicMutable[LayerNormSample, LayerNormSample]):

    def __init__(self,
                 max_embed_dim: int,
                 alias: Optional[str] = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

        self.max_embed_dim = max_embed_dim

        # warp layernorm
        self.layernorm = nn.LayerNorm(max_embed_dim)

        # store parameters
        self.samples: Dict[str, nn.Parameter] = {}
        # store args
        self._choice: LayerNormSample = LayerNormSample(max_embed_dim)

    def sample_parameters(self, choice: LayerNormSample) -> None:
        self._choice = choice
        self.samples['weight'] = self.layernorm.weight[:self._choice.embed_dim]
        self.samples['bias'] = self.layernorm.bias[:self._choice.embed_dim]

    def forward_all(self, x: Any) -> Any:
        max_choice = LayerNormSample(embed_dim=self.max_embed_dim)
        self.sample_parameters(max_choice)
        return F.layer_norm(
            x, (self._choice.embed_dim, ),
            weight=self.samples['weight'],
            bias=self.samples['bias'],
            eps=self.layernorm.eps)

    def forward_choice(self,
                       x: Any,
                       choice: Optional[LayerNormSample] = None) -> Any:
        if choice is not None:
            self.sample_parameters(choice)
            return F.layer_norm(
                x, (self._choice.embed_dim, ),
                weight=self.samples['weight'],
                bias=self.samples['bias'],
                eps=self.layernorm.eps)
        else:
            # assert already called sample_parameters
            assert self.samples is not None, \
                'Please call `sample_parameters` before forward_choice'
            return F.layer_norm(
                x, (self._choice.embed_dim, ),
                weight=self.samples['weight'],
                bias=self.samples['bias'],
                eps=self.layernorm.eps)

    def fix_chosen(self, chosen: LayerNormSample) -> None:
        """fix chosen"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLayerNorm is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        # new a layernorm layer
        temp_weight: nn.Parameter = self.layernorm.weight.data[:chosen.
                                                               embed_dim]
        temp_bias: nn.Parameter = self.layernorm.bias.data[:chosen.embed_dim]
        self.layernorm.weight = nn.Parameter(temp_weight)
        self.layernorm.bias = nn.Parameter(temp_bias)

        self._choice = chosen
        self.is_fixed = True

    def forward_fixed(self, x: Any) -> Any:
        return F.layer_norm(
            x, (self._choice.embed_dim, ),
            weight=self.layernorm.weight,
            bias=self.layernorm.bias,
            eps=self.layernorm.eps)

    def choices(self) -> List[LayerNormSample]:
        return super().choices

    def calc_sampled_flops(self, x: Any) -> float:
        return x * self._choice.embed_dim

    def calc_sampled_params(self) -> float:
        assert 'weight' in self.samples.keys(), \
            'Please call sample_parameters before calc params'
        assert 'bias' in self.samples.keys(), \
            'Please call sample_parameters before calc params'
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def sample_choice(self) -> LayerNormSample:
        """sample with random choice"""
        return LayerNormSample(random.randint(0, self.max_embed_dim))


class LayerNormSuper(LayerNorm):

    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(
            x, (self.sample_embed_dim, ),
            weight=self.samples['weight'],
            bias=self.samples['bias'],
            eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim
