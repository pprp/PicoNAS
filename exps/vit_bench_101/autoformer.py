import math
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class BaseTransformerModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for Transformer models.

    Attributes:
        - self.features (List[Tensor]): the features in each block.
        - self.feature_dims (List[int]): the dimension of features in each block.
        - self.distill_logits (Tensor|None): the logits of the distillation token, only for DeiT.
    """

    def __init__(self):
        super(BaseTransformerModel, self).__init__()
        # Base configs for Transformers
        self.img_size = 224
        self.patch_size = 16
        self.patch_stride = 16
        self.patch_padding = 0
        self.in_channels = 3
        self.hidden_dim = [192, 216, 240]
        self.depth = [12, 13, 14]
        self.num_heads = [3, 4]
        self.mlp_ratio = [3.5, 4.0]
        self.drop_rate = 0
        self.drop_path_rate = 0.1
        self.attn_drop_rate = 0

        # Calculate the dimension of features in each block
        if isinstance(self.hidden_dim, int):
            assert isinstance(self.depth, int)
            self.feature_dims = [self.hidden_dim] * self.depth
        elif isinstance(self.hidden_dim, (list, tuple)):
            assert isinstance(self.depth, (list, tuple))
            assert len(self.hidden_dim) == len(self.depth)
            self.feature_dims = sum(
                [[self.hidden_dim[i]] * d for i, d in enumerate(self.depth)], []
            )
        else:
            raise ValueError
        self.features = list()
        self.distill_logits = None

    def initialize_hooks(self, layers):
        """
        Initialize hooks for the given layers.
        """
        for layer in layers:
            layer.register_forward_hook(self._feature_hook)
        self.register_forward_pre_hook(lambda module, inp: self.features.clear())

    @abstractmethod
    def _feature_hook(self, module, inputs, outputs):
        pass

    def complexity(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'params': f'{round(params/1e6, 2)}M'}


def layernorm(w_in):
    return nn.LayerNorm(w_in, eps=1e-6)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        qkv_bias=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        qk_scale=None,
    ):
        super(MultiheadAttention, self).__init__()
        assert out_channels % num_heads == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.norm_factor = qk_scale if qk_scale else (out_channels // num_heads) ** -0.5
        self.qkv_transform = nn.Linear(in_channels, out_channels * 3, bias=qkv_bias)
        self.projection = nn.Linear(out_channels, out_channels)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        N, L, _ = x.shape
        x = (
            self.qkv_transform(x)
            .view(N, L, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = x[0], x[1], x[2]

        qk = query @ key.transpose(-1, -2) * self.norm_factor
        qk = F.softmax(qk, dim=-1)
        qk = self.attention_dropout(qk)

        out = qk @ value
        out = out.transpose(1, 2).contiguous().view(N, L, self.out_channels)
        out = self.projection(out)
        out = self.projection_dropout(out)

        if self.in_channels != self.out_channels:
            out = out + value.squeeze(1)

        return out


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0, hidden_ratio=1.0):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = int(in_channels * hidden_ratio)
        self.fc1 = nn.Linear(in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, out_channels)
        self.drop = nn.Dropout(drop_rate)
        self.activation_fn = torch.nn.GELU()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads,
        qkv_bias=False,
        out_channels=None,
        mlp_ratio=1.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        qk_scale=None,
    ):
        super(TransformerLayer, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = layernorm(in_channels)
        self.attn = MultiheadAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            qk_scale=qk_scale,
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )
        self.norm2 = layernorm(out_channels)
        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            drop_rate=drop_rate,
            hidden_ratio=mlp_ratio,
        )

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:
            x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, out_channels=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AutoFormerSub(BaseTransformerModel):
    def __init__(self, arch_config=None, num_classes=None):
        super(AutoFormerSub, self).__init__()
        # the configs of super arch

        if arch_config:
            self.num_heads = arch_config['num_heads']
            self.mlp_ratio = arch_config['mlp_ratio']
            self.hidden_dim = arch_config['hidden_dim']
            self.depth = arch_config['depth']

        else:
            self.num_heads = [3, 4]
            self.mlp_ratio = [3.5, 4.0]
            self.hidden_dim = [192, 216, 240]
            self.depth = [12, 13, 14]

        if num_classes:
            self.num_classes = num_classes
        else:
            self.num_classes = 100

        # print('hidden dim is:'. self.hidden_dim)
        self.feature_dims = [self.hidden_dim] * self.depth

        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.hidden_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = 1

        self.blocks = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)
        ]  # stochastic depth decay rule

        for i in range(self.depth):
            self.blocks.append(
                TransformerLayer(
                    in_channels=self.hidden_dim,
                    num_heads=self.num_heads[i],
                    qkv_bias=True,
                    mlp_ratio=self.mlp_ratio[i],
                    drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=dpr[i],
                )
            )

        self.initialize_hooks(self.blocks)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.num_tokens, self.hidden_dim)
        )
        trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = layernorm(self.hidden_dim)

        # classifier head
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

        self.apply(self._init_weights)

        self.distill_logits = None

        self.distill_token = None
        self.distill_head = None

    def _feature_hook(self, module, inputs, outputs):
        feat_size = int(self.patch_embed.num_patches**0.5)
        x = outputs[:, self.num_tokens :].view(
            outputs.size(0), feat_size, feat_size, self.hidden_dim
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        self.features.append(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.num_tokens == 1:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        else:
            x = torch.cat(
                [
                    self.cls_token.repeat(x.size(0), 1, 1),
                    self.distill_token.repeat(x.size(0), 1, 1),
                    x,
                ],
                dim=1,
            )

        x = x + self.pos_embed
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return torch.mean(x[:, 1:], dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        logits = self.head(x)
        if self.num_tokens == 1:
            return logits

        self.distill_logits = None
        self.distill_logits = self.distill_head(x)

        if self.training:
            return logits
        else:
            return (logits + self.distill_logits) / 2
