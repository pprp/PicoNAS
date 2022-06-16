import random
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..dynamic_mutable import DynamicMutable


class PatchSample(NamedTuple):
    sample_embed_dim: int


class DynamicPatchEmbed(DynamicMutable[PatchSample, PatchSample]):

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 max_embed_dim: int = 768,
                 scale: bool = False,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels,
            max_embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

        self.max_embed_dim = max_embed_dim
        self.scale = scale

        # store parameter
        self.samples: Dict[str, nn.Parameter] = {}
        # store args
        self._choice: PatchSample = PatchSample(self.max_embed_dim)

    def sample_choice(self) -> PatchSample:
        return PatchSample(random.randint(0, self.max_embed_dim))

    def sample_parameters(self, choice: PatchSample) -> None:
        self._choice = choice
        assert choice.sample_embed_dim <= self.max_embed_dim, \
            'Sampled embed dim should smaller or equal than max embed dim.'

        self.samples['weight'] = self.proj.weight[:self._choice.
                                                  sample_embed_dim, ...]
        self.samples['bias'] = self.proj.bias[:self._choice.sample_embed_dim,
                                              ...]
        if self.scale:
            self.samples[
                'scale'] = self.max_embed_dim / self._choice.sample_embed_dim

    def forward_all(self, x: Tensor) -> Tensor:
        max_choice = PatchSample(self.max_embed_dim)
        self.sample_parameters(max_choice)
        x = F.conv2d(
            x,
            self.samples['weight'],
            self.samples['bias'],
            stride=self.patch_size,
            padding=self.proj.padding,
            dilation=self.proj.dilation).flatten(2).transpose(1, 2)
        if self.scale:
            return x * self.samples['scale']
        return x

    def forward_choice(self,
                       x: Tensor,
                       choice: Optional[PatchSample] = None) -> Tensor:
        if choice is not None:
            self.sample_parameters(choice)
            x = F.conv2d(
                x,
                self.samples['weight'],
                self.samples['bias'],
                stride=self.patch_size,
                padding=self.proj.padding,
                dilation=self.proj.dilation).flatten(2).transpose(1, 2)
            if self.scale:
                return x * self.samples['scale']
            return x
        else:
            # chose the lagest
            assert self.samples is not None, \
                'Please call `sample_parameters` before forward_choice'
            x = F.conv2d(
                x,
                self.samples['weight'],
                self.samples['bias'],
                stride=self.patch_size,
                padding=self.proj.padding,
                dilation=self.proj.dilation).flatten(2).transpose(1, 2)
            if self.scale:
                return x * self.samples['scale']
            return x

    def fix_chosen(self, chosen: PatchSample) -> None:
        """fix chosen"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of DynamicLinear is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        # new a conv2d layer
        temp_weight = self.proj.weight.data[:chosen.sample_embed_dim, ...]
        temp_bias = self.proj.bias.data[:chosen.sample_embed_dim]
        self.proj.weight = nn.Parameter(temp_weight)
        self.proj.bias = nn.Parameter(temp_bias)

        self._choice = chosen
        self.is_fixed = True

    def forward_fixed(self, x: Tensor) -> Tensor:
        x = F.conv2d(
            x,
            self.proj.weight,
            self.proj.bias,
            stride=self.patch_size,
            padding=self.proj.padding,
            dilation=self.proj.dilation).flatten(2).transpose(1, 2)
        if self.scale:
            return x * self.samples['scale']
        return x

    def choices(self) -> List[PatchSample]:
        return super().choices


class PatchembedSuper(DynamicMutable):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 scale=False):
        super().__init__()

        img_size = tuple(img_size, img_size)
        patch_size = tuple(patch_size, patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = embed_dim
        self.scale = scale

        # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model" \
            f' ({self.img_size[0]}*{self.img_size[1]}).'
        x = F.conv2d(
            x,
            self.sampled_weight,
            self.sampled_bias,
            stride=self.patch_size,
            padding=self.proj.padding,
            dilation=self.proj.dilation).flatten(2).transpose(1, 2)
        if self.scale:
            return x * self.sampled_scale
        return x

    def calc_sampled_param_num(self):
        return self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
            total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops
