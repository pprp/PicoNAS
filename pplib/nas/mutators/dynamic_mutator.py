# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Type

from torch.nn import Module

from pplib.nas.mutables.dynamic_mixin import DynamicMixin
from pplib.nas.mutables.mutable_value import MutableValue
from pplib.nas.mutators.base_mutator import MUTABLE_TYPE, BaseMutator


class DynamicMutator(BaseMutator):
    """_summary_

    Args:
        BaseMutator (_type_): _description_

    Example:
        mutator.prepare_from_supernet(supernet)
        mutator.sample_value(mode='max')
        mutator.fix_chosen(supernet)

    """

    def __init__(
        self,
        custom_group: Optional[List[List[str]]] = None,
        with_alias: bool = False,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)

        if custom_group is None:
            custom_group = []
        self._custom_group = custom_group

        self._search_group: Dict[int, List[MUTABLE_TYPE]] = {}
        self._with_alias = with_alias

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        global_count = 0
        for _, module in supernet.named_modules():
            if isinstance(module, self.mutable_class_type):
                attrs = vars(module)
                for v in attrs.values():
                    if isinstance(v, MutableValue):
                        self._search_group[global_count] = v
                        global_count += 1

    def merge_search_group(self):
        """merge search group with same mutable value"""
        pass

    def sample_value(self, mode: str = 'max') -> Any:
        """Tranverse all of DynamicMutable and change value by mode."""
        for id, mutable in self._search_group.items():
            mutable.sample_value(mode)

    def fix_chosen(self, supernet) -> None:
        """Tranverse all of Mutable and fix chosen operations."""
        for _, module in supernet.named_modules():
            if isinstance(module, self.mutable_class_type):
                module.fix_chosen()

    @property
    def mutable_class_type(self) -> Type[MUTABLE_TYPE]:
        """Corresponding mutable class type.

        Returns:
            Type[MUTABLE_TYPE]: Mutable class type.
        """
        return DynamicMixin

    @property
    def search_group(self) -> Dict[int, List[MUTABLE_TYPE]]:
        """Search group of supernet.

        Note:
            For mutable based mutator, the search group is composed of
            corresponding mutables.

        Raises:
            RuntimeError: Called before search group has been built.

        Returns:
            Dict[int, List[MUTABLE_TYPE]]: Search group.
        """
        if self._search_group is None:
            raise RuntimeError(
                'Call `prepare_from_supernet` before access search group!')
        return self._search_group
