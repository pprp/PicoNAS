# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Type, TypeVar

from mmcv.runner import BaseModule
from torch.nn import Module

from ..mutables.base_mutable import BaseMutable


class BaseMutator(ABC, BaseModule):
    """The base class for mutator.

    Mutator is mainly used for subnet management, it usually provides functions
    such as sampling and setting of subnets.

    All subclasses should implement the following APIs:

    - ``prepare_from_supernet()``
    - ``search_space``

    Args:
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

    @abstractmethod
    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """

    @property
    @abstractmethod
    def search_group(self) -> Dict:
        """Search group of the supernet.

        Note:
            Search group is different from search space. The key of search
            group is called ``group_id``, and the value is corresponding
            searchable modules. The searchable modules will have the same
            search space if they are in the same group.

        Returns:
            dict: Search group.
        """


MUTABLE_TYPE = TypeVar('MUTABLE_TYPE', bound=BaseMutable)


class ArchitectureMutator(BaseMutator, Generic[MUTABLE_TYPE]):
    """The base class for mutable based mutator.

    All subclass should implement the following APIS:

    - ``mutable_class_type``

    Args:
        custom_group (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
        with_alias (bool): whether use alias to generate search groups.
            Default to False.
    """

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

    # TODO
    # should be a class property
    @property
    @abstractmethod
    def mutable_class_type(self) -> Type[MUTABLE_TYPE]:
        """Corresponding mutable class type.

        Returns:
            Type[MUTABLE_TYPE]: Mutable class type.
        """

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Note:
            For mutable based mutator, we need to build search group first.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        return self._build_search_group(supernet)

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

    def _build_search_group(self, supernet: Module) -> None:
        """Build search group with ``supernet`` and ``custom_group``.

        Note:
            Apart from user-defined search group, all other searchable
            modules(mutable) will be grouped separately.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        if self._with_alias:
            self._build_search_group_by_alias(supernet)
        else:
            self._build_search_group_by_module_name(supernet)

    def _build_search_group_by_module_name(self, supernet: Module) -> None:
        """Build search group with ``supernet`` and ``custom_group``.

        Note:
            Apart from user-defined search group, all other searchable
            modules(mutable) will be grouped separately.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        module_name2module: Dict[str, Module] = dict()
        for name, module in supernet.named_modules():
            module_name2module[name] = module

        # Map module to group id for user-defined group
        self.module_name2group_id: Dict[str, int] = dict()
        for idx, group in enumerate(self._custom_group):
            for module_name in group:
                assert module_name in module_name2module, \
                    f'`{module_name}` is not a module name of supernet, ' \
                    f'expected module names: {module_name2module.keys()}'

                if module_name in module_name2module:
                    self.module_name2group_id[module_name] = idx

        search_group: Dict[int, List[MUTABLE_TYPE]] = dict()
        current_group_nums = len(self._custom_group)

        for name, module in module_name2module.items():
            if isinstance(module, self.mutable_class_type):
                group_id = self.module_name2group_id.get(name)

                if group_id is None:
                    group_id = current_group_nums
                    current_group_nums += 1
                try:
                    search_group[group_id].append(module)
                except KeyError:
                    search_group[group_id] = [module]
                self.module_name2group_id[name] = group_id
        self._search_group = search_group

    def _build_search_group_by_alias(self, supernet: Module) -> None:
        """Build search group with ``supernet`` and ``custom_group``.

        Note:
            Apart from user-defined search group, all other searchable
            modules(mutable) will be grouped separately.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        alias2module: Dict[str, Module] = dict()
        for name, module in supernet.named_modules():
            if isinstance(module, self.mutable_class_type):
                if module.alias is not None:
                    if module.alias not in alias2module:
                        alias2module[module.alias] = [module]
                    else:
                        alias2module[module.alias].append(module)

        # Map module to group id for user-defined group
        self.alias2group_id: Dict[str, int] = dict()
        for idx, group in enumerate(self._custom_group):
            for module_name in group:
                assert module_name in alias2module, \
                    f'`{module_name}` is not a module name of supernet, ' \
                    f'expected module names: {alias2module.keys()}'

                if module_name in alias2module:
                    self.alias2group_id[module_name] = idx

        search_group: Dict[int, List[MUTABLE_TYPE]] = dict()

        self.alias2group_id_copy = self.alias2group_id.copy()
        for gid, (alias_name, modules) in enumerate(alias2module.items()):

            for module in modules:
                if len(self.alias2group_id_copy.keys()) > 0:
                    group_id = self.alias2group_id_copy.get(alias_name)
                    assert group_id is not None, \
                        f'Get wrong alias name: {alias_name}'
                else:
                    group_id = gid

                try:
                    search_group[group_id].append(module)
                except KeyError:
                    search_group[group_id] = [module]

                self.alias2group_id[alias_name] = group_id

        self._search_group = search_group
