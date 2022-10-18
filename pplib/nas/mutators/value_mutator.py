from typing import Dict, List, Optional, Type

from torch.nn import Module

from pplib.nas.mutables import MutableValue
from .base_mutator import ArchitectureMutator


class ValueMutator(ArchitectureMutator[MutableValue]):
    """The base class for mutable based mutator.
    All subclass should implement the following APIS:
    - ``mutable_class_type``
    Args:
        custom_group (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
    """

    def __init__(self,
                 custom_group: Optional[List[List[str]]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)

        if custom_group is None:
            custom_group = []
        self._custom_group = custom_group
        self._search_groups: Optional[Dict[int, List[MutableValue]]] = None

    @property
    def mutable_class_type(self) -> Type[MutableValue]:
        """Corresponding mutable class type.
        Returns:
            Type[MUTABLE_TYPE]: Mutable class type.
        """
        return MutableValue

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.
        Note:
            For mutable based mutator, we need to build search group first.
        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        self._build_search_groups(supernet)

    @property
    def search_groups(self) -> Dict[int, List[MutableValue]]:
        """Search group of supernet.
        Note:
            For mutable based mutator, the search group is composed of
            corresponding mutables.
        Raises:
            RuntimeError: Called before search group has been built.
        Returns:
            Dict[int, List[MUTABLE_TYPE]]: Search group.
        """
        if self._search_groups is None:
            raise RuntimeError(
                'Call `prepare_from_supernet` before access search group!')
        return self._search_groups

    def _build_search_groups(self, supernet: Module) -> None:
        self._search_groups = self.group_by_name_and_alias(
            supernet, self._custom_group)
