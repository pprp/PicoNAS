from typing import Dict, NamedTuple, Optional

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiheadAttention

from pplib.nas.mutables.dynamic_mutable import DynamicMutable
from .dynamic_linear import LinearSuper
from .dynamic_qkv import qkv_super
from .dynamic_relativeposition import RelativePosition2D_super


class AttentionSample(NamedTuple):
    sample_q_embed_dim: int
    sample_num_heads: int
    sample_in_embed_dim: int

# torch 的 MHA 中逻辑过于复杂，这里是基于 mmcls 的 MHA 开发的


class DynamicMHA(MultiheadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # auto former 的官方实现中 head dims 是固定的 64
        # 这里的实现是 head dims 可搜索
        self.mutable_head_dims = OrderChannelMutable(self.head_dims, candidate_choices=[
                                                     int(0.5 * self.head_dims), self.head_dims])
        self.mutable_embed_dims = OrderChannelMutable(
            self.embed_dims, candidate_choices=[320, 384, 448])
        # 还没设计实现 MutableValue，先拿一个 list 简单测试一下
        self.mutable_num_heads = [5, 6, 7]
        # TODO del, just for test
        self.mutable_head_dims.current_choice = int(0.5 * self.head_dims)

    def _get_q_out_mask(self):
        # TODO sample, min is just for test.
        active_num_heads = min(self.mutable_num_heads)
        active_heads_mask = torch.cat(
            [self.mutable_head_dims.mask] * active_num_heads, dim=0)

        inactive_num_heads = max(self.mutable_num_heads) - active_num_heads
        inactive_mask = torch.zeros_like(self.mutable_head_dims.mask).bool()
        inactive_heads_mask = torch.cat(
            [inactive_mask] * inactive_num_heads, dim=0)

        q_out_mask = torch.cat([active_heads_mask, inactive_heads_mask], dim=0)

        return q_out_mask

    def _get_qkv_weight_bias(self):
        q_out_mask = self._get_q_out_mask()
        out_mask = torch.cat([q_out_mask] * 3, dim=0)
        in_mask = self.mutable_embed_dims.mask

        weight = self.qkv.weight[out_mask][:, in_mask]
        bias = self.qkv.bias[out_mask] if self.qkv.bias is not None else None
        return weight, bias

    def _get_proj_weight_bias(self):
        out_mask = self.mutable_embed_dims.mask
        in_mask = self._get_q_out_mask()

        weight = self.proj.weight[out_mask][:, in_mask]
        bias = self.proj.bias[out_mask] if self.qkv.bias is not None else None
        return weight, bias

    def forward(self, x):
        B, N, _ = x.shape

        qkv_weight, qkv_bias = self._get_qkv_weight_bias()
        qkv = F.linear(x, qkv_weight, qkv_bias)

        # TODO mutable value, min is just for test
        current_num_heads = min(self.mutable_num_heads)

        current_head_dims = self.mutable_head_dims.mask.sum()
        current_embed_dims = self.mutable_embed_dims.mask.sum()
        qkv = qkv.reshape(B, N, 3, current_num_heads,
                          current_head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N,
                                               current_num_heads * current_head_dims)
        proj_weight, proj_bias = self._get_proj_weight_bias()
        x = F.linear(x, proj_weight, proj_bias)

        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
