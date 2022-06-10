from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear

from .oneshot_mutable import OneShotMutable
from .utils import trunc_normal_


class DynamicMutable(OneShotMutable[Any, Any]):
    """_summary_


    Note: autoformer -> ours
        sample_parameters -> sample_parameters
        set_sample_config -> set_forward_args
        calc_sampled_param_num -> calc_sampled_params
        get_complexity -> calc_sampled_flops

    Args:
        module_kwargs (Optional[Dict[str, Dict]], optional): _description_.
            Defaults to None.
        alias (Optional[str], optional): _description_. Defaults to None.
        init_cfg (Optional[Dict], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    def __init__(self,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

    @abstractmethod
    def sample_parameters(self, choice: Dict) -> None:
        """Modify the sample property. This function would be called in
        `modify_forward` function.

        Args:
            choice (Dict): _description_
        """

    @abstractmethod
    def calc_sampled_params(self) -> float:
        """calculate the parameter of sampled mutable"""

    @abstractmethod
    def calc_sampled_flops(self) -> float:
        """calculate the FLOPs of sampled mutable"""

    def set_forward_args(self, choice: Dict) -> None:
        """Interface for modifying the choice using partial"""
        return super().set_forward_args(choice)

    @abstractmethod
    def fix_chosen(self, chosen: Dict) -> None:
        return super().fix_chosen(chosen)

    @abstractmethod
    def sample_choice(self) -> Dict:
        """sample choice on dynamic mutable"""

    # @abstractmethod
    # def forward_fixed(self, x: Any) -> Any:
    #     """Forward when the mutable is fixed.

    #     All subclasses must implement this method.
    #     """

    # @abstractmethod
    # def forward_all(self, x: Any) -> Any:
    #     """Forward all choices."""

    # @abstractmethod
    # def forward_choice(self,
    #                    x: Any,
    #                    choice: Optional[Any] = None) -> Any:
    #     """Forward when the mutable is not fixed.

    #     All subclasses must implement this method.
    #     """


class SlimmableLinear(OneShotMutable[int, int], Linear):

    def __init__(self,
                 in_features_list: List[int],
                 out_features_list: List[int],
                 bias=True) -> None:
        Linear.__init__(
            in_featuress=max(in_features_list),
            out_features=max(out_features_list),
            bias=bias)
        assert len(self.in_features_list) == len(self.out_features_list)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self._chosen: Optional[int] = None
        self._is_fixed = False

    def forward_fixed(self, x) -> Tensor:
        assert self._is_fixed is True
        assert self._chosen is not None
        self.in_features = self.in_features_list[self._chosen]
        self.out_features = self.out_features_list[self._chosen]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias

        return nn.functional.linear(x, weight, bias)

    def forward_choice(self,
                       x: Tensor,
                       choice: Optional[int] = None) -> Tensor:
        self.fix_chosen(choice)
        return self.forward_fixed(x)

    def fix_chosen(self, chosen: int) -> None:
        """chosen index"""
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        self._chosen = chosen
        self.is_fixed = True

    def choices(self) -> List[int]:
        """index list"""
        return list(range(len(self.in_features_list)))


class LinearSuper(OneShotMutable, Linear):
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

        self.samples = {}

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


class PatchembedSuper(OneShotMutable):

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


class RelativePosition2D_super(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical
        #     embedding for the class
        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

        self.sample_head_dim = None
        self.sample_embeddings_table_h = None
        self.sample_embeddings_table_v = None

    def set_sample_config(self, sample_head_dim):
        self.sample_head_dim = sample_head_dim
        self.sample_embeddings_table_h = \
            self.embeddings_table_h[:, :sample_head_dim]
        self.sample_embeddings_table_v = \
            self.embeddings_table_v[:, :sample_head_dim]

    def calc_sampled_param_num(self):
        return self.sample_embeddings_table_h.numel(
        ) + self.sample_embeddings_table_v.numel()

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (
            range_vec_k[None, :] // int(length_q**0.5) -
            range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (
            range_vec_k[None, :] % int(length_q**0.5) -
            range_vec_q[:, None] % int(length_q**0.5))
        # clip the distance to the range of
        #      [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1],
        #      0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              'constant', 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v).cuda()
        final_mat_h = torch.LongTensor(final_mat_h).cuda()
        # get the embeddings with the corresponding distance
        embeddings = self.sample_embeddings_table_v[
            final_mat_v] + self.sample_embeddings_table_h[final_mat_h]

        return embeddings


class AttentionSuper(nn.Module):

    def __init__(self,
                 super_embed_dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 normalization=False,
                 relative_position=False,
                 num_patches=None,
                 max_relative_position=14,
                 scale=False,
                 change_qkv=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.super_embed_dim = super_embed_dim

        self.fc_scale = scale
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(
                super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)
        else:
            self.qkv = LinearSuper(
                super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position)
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None

        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(self,
                          sample_q_embed_dim=None,
                          sample_num_heads=None,
                          sample_in_embed_dim=None):

        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (sample_in_embed_dim //
                                 self.sample_num_heads)**-0.5

        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim //
                                 self.sample_num_heads)**-0.5

        self.qkv.set_sample_config(
            sample_in_dim=sample_in_embed_dim,
            sample_out_dim=3 * self.sample_qk_embed_dim)
        self.proj.set_sample_config(
            sample_in_dim=self.sample_qk_embed_dim,
            sample_out_dim=sample_in_embed_dim)
        if self.relative_position:
            self.rel_pos_embed_k.set_sample_config(self.sample_qk_embed_dim //
                                                   sample_num_heads)
            self.rel_pos_embed_v.set_sample_config(self.sample_qk_embed_dim //
                                                   sample_num_heads)

    def calc_sampled_param_num(self):

        return 0

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.qkv.get_complexity(sequence_length)
        # attn
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim  # noqa: E501
        # x
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim  # noqa: E501
        total_flops += self.proj.get_complexity(sequence_length)
        if self.relative_position:
            total_flops += self.max_relative_position * sequence_length * \
                sequence_length + sequence_length * sequence_length / 2.0
            total_flops += self.max_relative_position * sequence_length * \
                sequence_length + sequence_length * self.sample_qk_embed_dim / 2.0  # noqa: E501
        return total_flops

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.sample_num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.sample_scale
        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.sample_num_heads * B, -1) @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.sample_num_heads, N, N) * self.sample_scale  # noqa: E501

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1,
                                  3).reshape(N, B * self.sample_num_heads, -1)
            # The size of attention is (B, num_heads, N, N), reshape it to
            # (N, B*num_heads, N) and do batch matmul with the relative
            # position embedding of V (N, N, head_dim) get shape like
            # (N, B*num_heads, head_dim). We reshape it to the same size
            # as x (B, num_heads, N, hidden_dim)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(
                B, self.sample_num_heads, N, -1).transpose(2, 1).reshape(
                    B, N, -1)

        if self.fc_scale:
            x = x * (self.super_embed_dim / self.sample_qk_embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
