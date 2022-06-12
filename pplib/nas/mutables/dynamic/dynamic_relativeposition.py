from typing import Any, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pplib.nas.mutables.dynamic_mutable import DynamicMutable
from ..utils import trunc_normal_


class RelativePositonSample(NamedTuple):
    sample_head_dim: int


class DynamicRelativePosion2D(DynamicMutable):
    """_summary_

    Args:
        num_units (int): _description_
        max_relative_position (int, optional): _description_. Defaults to 14.
        alias (Optional[str], optional): _description_. Defaults to None.
        module_kwargs (Optional[Dict[str, Dict]], optional): _description_.
            Defaults to None.
        init_cfg (Optional[Dict], optional): _description_. Defaults to None.
    """

    def __init__(self,
                 num_units: int,
                 max_relative_position: int = 14,
                 alias: Optional[str] = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

        self.num_units = num_units
        self.max_relative_position = max_relative_position

        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

        # store parameters
        self.samples: Dict[str, nn.Parameter] = {}
        # store args
        self._choices: RelativePositonSample = RelativePositonSample(
            self.num_units)

    def sample_parameters(self, choice: RelativePositonSample) -> None:
        self._choices = choice
        self.samples['embeddings_table_v'] = \
            self.embeddings_table_v[:, :self._choices.sample_head_dim]
        self.samples['embeddings_table_h'] = \
            self.embeddings_table_h[:, :self._choices.sample_head_dim]

    def forward_all(self, x: Dict) -> Tensor:
        length_q = x['length_q']
        length_k = x['length_k']
        max_choice = RelativePositonSample(self.max_relative_position)
        self.sample_parameters(max_choice)

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
        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), 'constant', 0)
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), 'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v).cuda()
        final_mat_h = torch.LongTensor(final_mat_h).cuda()
        # get the embeddings with the corresponding distance
        embeddings = self.samples['embeddings_table_v'][
            final_mat_v] + self.samples['embeddings_table_h'][final_mat_h]

        return embeddings

    def forward_choice(
            self,
            x: Tensor,
            choice: Optional[RelativePositonSample] = None) -> Tensor:
        return super().forward_choice(x, choice)

    def fix_chosen(self, chosen: RelativePositonSample) -> None:
        return super().fix_chosen(chosen)

    def forward_fixed(self, x: Tensor) -> Tensor:
        return super().forward_fixed(x)

    def calc_sampled_flops(self, x: Any) -> float:
        return super().calc_sampled_flops(x)

    def calc_sampled_params(self) -> float:
        return super().calc_sampled_params()

    def sample_choice(self) -> RelativePositonSample:
        return super().sample_choice()

    def choices(self) -> List[RelativePositonSample]:
        return super().choices


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
        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), 'constant', 0)
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), 'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v).cuda()
        final_mat_h = torch.LongTensor(final_mat_h).cuda()
        # get the embeddings with the corresponding distance
        embeddings = self.sample_embeddings_table_v[
            final_mat_v] + self.sample_embeddings_table_h[final_mat_h]

        return embeddings
