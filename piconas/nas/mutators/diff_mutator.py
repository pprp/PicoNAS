# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn

from piconas.nas.mutables import DiffMutable
from .base_mutator import ArchitectureMutator


class DiffMutator(ArchitectureMutator[DiffMutable]):
    """Differentiable mutable based mutator.

    `DiffMutator` is responsible for mutating `DiffMutable`, `DiffOP`,
    `DiffChoiceRoute` and `GumbelChoiceRoute`. The architecture
    parameters of the mutables are retained in `DiffMutator`.

    Args:
        custom_group (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
        with_alias (bool): whether use alias to generate search groups.
            Default to False.
    """

    def __init__(
        self,
        custom_group: Optional[List[List[str]]] = None,
        with_alias: bool = False,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            custom_group=custom_group, with_alias=with_alias, init_cfg=init_cfg
        )

    def build_arch_param(self, num_choices) -> nn.Parameter:
        """Build learnable architecture parameters."""
        return nn.Parameter(torch.randn(num_choices) * 1e-3)

    def prepare_from_supernet(self, supernet: nn.Module) -> None:
        """Inherit from ``BaseMutator``'s, generate `arch_params` in DARTS.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """

        super().prepare_from_supernet(supernet)
        self.arch_params = self.build_arch_params()
        self.modify_supernet_forward(self.arch_params)

    def build_arch_params(self):
        """This function will build many arch params, which are generally used
        in differentiable search algorithms, such as Darts' series. Each
        group_id corresponds to an arch param, so the Mutables with the same
        group_id share the same arch param.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        Returns:
            torch.nn.ParameterDict: the arch params are got after traversing
                the supernet.
        """

        arch_params = nn.ParameterDict()
        for group_id, modules in self.search_group.items():
            group_arch_param = self.build_arch_param(modules[0].num_choices)
            arch_params[str(group_id)] = group_arch_param
        return arch_params

    def modify_supernet_forward(self, arch_params):
        """Modify the DiffMutable's default arch_param in forward.

        The `arch_param` is along to `DiffMutator`, while the
        `DiffMutable` needs `arch_param` in the forward. Here we use partial
        function to assign the corresponding `arch_param` to each
        `DiffMutable`.
        """
        for group_id, mutables in self.search_group.items():
            for m in mutables:
                m.set_forward_args(arch_param=arch_params[str(group_id)])

    @property
    def mutable_class_type(self) -> Type[DiffMutable]:
        """Differentiable mutable class type.

        Returns:
            Type[DiffMutable]: Class type of differentiable mutable.
        """
        return DiffMutable

    def sample_choices(self):
        """Sampling by search groups.
        The sampling result of the first mutable of each group is the sampling
        result of this group.
        Returns:
            Dict[int, Any]: Random choices dict.
        """
        choices = dict()
        for group_id, mutables in self.search_group.items():
            arch_param = self.arch_params[str(group_id)]
            if mutables[0].is_fixed:
                choice = mutables[0]._chosen[0]
            else:
                choice = mutables[0].sample_choice(arch_param)
            choices[group_id] = choice
        return choices

    def set_choices(self, choices: Dict[int, Any]) -> None:
        """Set mutables' current choice according to choices sample by
        :func:`sample_choices`.
        Args:
            choices (Dict[int, Any]): Choices dict. The key is group_id in
                search groups, and the value is the sampling results
                corresponding to this group.
        """
        for group_id, mutables in self.search_group.items():
            choice = choices[group_id]
            for m in mutables:
                m.current_choice = choice
