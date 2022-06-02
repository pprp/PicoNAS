# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from mmcv.runner import BaseModule

CHOICE_TYPE = TypeVar('CHOICE_TYPE')

CHOSEN_TYPE = TypeVar('CHOSEN_TYPE')


class BaseMutable(BaseModule, ABC, Generic[CHOICE_TYPE, CHOSEN_TYPE]):
    """Base Class for mutables. Mutable means a searchable module widely used
    in Neural Architecture Search(NAS).

    It mainly consists of some optional operations, and achieving
    searchable function by handling choice with ``MUTATOR``.

    All subclass should implement the following APIs:

    - ``forward()``
    - ``fix_chosen()``
    - ``choices()``

    Args:
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.module_kwargs = module_kwargs
        self.alias = alias
        self._is_fixed = False

    @property
    def is_fixed(self) -> bool:
        """bool: whether the mutable is fixed.

        Note:
            If a mutable is fixed, it is no longer a searchable module, just
                a normal fixed module.
            If a mutable is not fixed, it still is a searchable module.
        """
        return self._is_fixed

    @is_fixed.setter
    def is_fixed(self, is_fixed: bool) -> None:
        """Set the status of `is_fixed`."""
        assert isinstance(is_fixed, bool), \
            f'The type of `is_fixed` need to be bool type, ' \
            f'but got: {type(is_fixed)}'
        if self._is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not set `is_fixed` function repeatedly.')
        self._is_fixed = is_fixed

    @property
    @abstractmethod
    def choices(self) -> List[CHOICE_TYPE]:
        """list: all choices.  All subclasses must implement this method."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward computation."""

    @abstractmethod
    def fix_chosen(self, chosen: CHOSEN_TYPE) -> None:
        """Fix mutable with choice. This function would fix the choice of
        Mutable. The :attr:`is_fixed` will be set to True and only the selected
        operations can be retained. All subclasses must implement this method.

        Note:
            This operation is irreversible.
        """

    @property
    def num_choices(self) -> int:
        """int: length of choices.
        """
        return len(self.choices)
