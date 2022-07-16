# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

from torch.nn import Module

from pplib.nas.mutables.dynamic_mutable import DynamicMutable
from pplib.nas.mutables.mutable_value import MutableValue
from pplib.nas.mutators.base_mutator import MUTABLE_TYPE, BaseMutator


class DynamicMutator(BaseMutator):

    def __init__(self,
                 custom_group: Optional[List[List[str]]] = None,
                 with_alias: bool = False,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)

        if custom_group is None:
            custom_group = []
        self._custom_group = custom_group

        self._search_group: Optional[Dict[int, List[MUTABLE_TYPE]]] = None
        self._with_alias = with_alias

    @abstractmethod
    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        global_count = 0
        for name, module in supernet.named_modules():
            if isinstance(module, DynamicMutable):
                attrs = vars(module)
                for k, v in attrs.items():
                    if isinstance(v, MutableValue):
                        self._search_group[global_count] = v 
                        global_count += 1 
                
    def sample_value(self, value: DynamicMutable, mode: str = 'max') -> Any:
        """Tranverse all of DynamicMutable and change value by mode."""

    @property
    def mutable_class_type(self) -> Type[MUTABLE_TYPE]:
        """Corresponding mutable class type.

        Returns:
            Type[MUTABLE_TYPE]: Mutable class type.
        """
        return DynamicMutable
