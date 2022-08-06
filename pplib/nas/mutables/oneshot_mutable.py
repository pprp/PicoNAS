# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_mutable import CHOICE_TYPE, CHOSEN_TYPE, BaseMutable


class OneShotMutable(BaseMutable[CHOICE_TYPE, CHOSEN_TYPE]):
    """Base class for one shot mutables.

    Args:
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.

    Note:
        :meth:`forward_all` is called when calculating FLOPs.
    """

    def __init__(
        self,
        module_kwargs: Optional[Dict[str, Dict]] = None,
        alias: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

    def forward(self, x: Any, choice: Optional[CHOICE_TYPE] = None) -> Any:
        """Calls either :func:`forward_fixed` or :func:`forward_choice`
        depending on whether :func:`is_fixed` is ``True``.

        Note:
            :meth:`forward_fixed` is called when in `fixed` mode.
            :meth:`forward_choice` is called when in `unfixed` mode.

        Args:
            x (Any): input data for forward computation.
            choice (CHOICE_TYPE, optional): the chosen key in ``MUTABLE``.

        Returns:
            Any: the result of forward
        """
        if self.is_fixed:
            return self.forward_fixed(x)
        else:
            return self.forward_choice(x, choice=choice)

    @property
    def random_choice(self) -> CHOICE_TYPE:
        """Sample random choice during searching.

        Returns:
            CHOICE_TYPE: the chosen key in ``MUTABLE``.
        """

    @abstractmethod
    def forward_fixed(self, x: Any) -> Any:
        """Forward when the mutable is fixed.

        All subclasses must implement this method.
        """

    @abstractmethod
    def forward_all(self, x: Any) -> Any:
        """Forward all choices."""

    @abstractmethod
    def forward_choice(self,
                       x: Any,
                       choice: Optional[CHOICE_TYPE] = None) -> Any:
        """Forward when the mutable is not fixed.

        All subclasses must implement this method.
        """

    def set_forward_args(self, choice: CHOICE_TYPE) -> None:
        """Interface for modifying the choice using partial."""
        forward_with_default_args: Callable[[Any, Optional[CHOICE_TYPE]],
                                            Any] = partial(
                                                self.forward,
                                                choice=choice)  # noqa:E501
        setattr(self, 'forward', forward_with_default_args)


class OneShotOP(OneShotMutable[str, str]):
    """A type of ``MUTABLES`` for single path supernet, such as Single Path One
    Shot. In single path supernet, each choice block only has one choice
    invoked at the same time. A path is obtained by sampling all the choice
    blocks.

    Args:
        candidate_ops (dict[str, dict]): the configs for the candidate
            operations.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(
        self,
        candidate_ops: nn.ModuleDict,
        module_kwargs: Optional[Dict[str, Dict]] = None,
        alias: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)
        assert len(candidate_ops) >= 1, (
            f'Number of candidate op must greater than 1, '
            f'but got: {len(candidate_ops)}')

        self._is_fixed = False
        self._chosen: Optional[str] = None
        self._candidate_ops = self._build_ops(candidate_ops)

    @staticmethod
    def _build_ops(candidate_ops: nn.ModuleDict) -> nn.ModuleDict:
        """Build candidate operations based on choice configures.

        Args:
            candidate_ops (dict[str, dict] | :obj:`nn.ModuleDict`): the configs
                for the candidate operations or nn.ModuleDict.
            module_kwargs (dict[str, dict], optional): Module initialization
                named arguments.

        Returns:
            ModuleDict (dict[str, Any], optional):  the key of ``ops`` is
                the name of each choice in configs and the value of ``ops``
                is the corresponding candidate operation.
        """
        if isinstance(candidate_ops, nn.ModuleDict):
            return candidate_ops
        else:
            raise NotImplementedError

    def forward_fixed(self, x: Any) -> Tensor:
        """Forward when the mutable is in `fixed` mode.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward the fixed operation.
        """
        return self._candidate_ops[self._chosen](x)

    def forward_choice(self, x: Any, choice: Optional[str] = None) -> Tensor:
        """Forward when the mutable is in `unfixed` mode.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
            choice (str, optional): the chosen key in ``MUTABLE``.

        Returns:
            Tensor: the result of forward the ``choice`` operation.
        """
        if choice is None:
            return self.forward_all(x)
        else:
            return self._candidate_ops[choice](x)

    def forward_all(self, x: Any) -> Tensor:
        """Forward all choices. Used to calculate FLOPs.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward all of the ``choice`` operation.
        """
        outputs = [op(x) for op in self._candidate_ops.values()]
        return sum(outputs)

    def fix_chosen(self, chosen: List[str]) -> None:
        """Fix mutable with subnet config. This operation would convert
        `unfixed` mode to `fixed` mode. The :attr:`is_fixed` will be set to
        True and only the selected operations can be retained.

        Args:
            chosen (str): the chosen key in ``MUTABLE``. Defaults to None.
        """
        if isinstance(chosen, str):
            chosen = [chosen]

        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        for c in self.choices:
            if c not in chosen:
                self._candidate_ops.pop(c)

        self._chosen = chosen
        self.is_fixed = True

    @property
    def random_choice(self) -> str:
        """uniform sampling."""
        return np.random.choice(self.choices, 1)[0]

    @property
    def choices(self) -> List[str]:
        """list: all choices."""
        return list(self._candidate_ops.keys())

    def set_forward_args(self, choice: CHOICE_TYPE) -> None:
        """Interface for modifying the choice using partial."""
        forward_with_default_args: Callable[[Any, Optional[CHOICE_TYPE]],
                                            Any] = partial(
                                                self.forward,
                                                choice=choice)  # noqa:E501
        setattr(self, 'forward', forward_with_default_args)


class OneShotProbOP(OneShotOP):
    """Sampling candidate operation according to probability.

    Args:
        candidate_ops (dict[str, dict]): the configs for the candidate
            operations.
        choice_probs (list): the probability of sampling each
            candidate operation.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(
        self,
        candidate_ops: Dict[str, Dict],
        choice_probs: list = None,
        module_kwargs: Optional[Dict[str, Dict]] = None,
        alias: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            candidate_ops=candidate_ops,
            module_kwargs=module_kwargs,
            alias=alias,
            init_cfg=init_cfg,
        )
        assert choice_probs is not None
        assert (sum(choice_probs) - 1 < np.finfo(np.float64).eps
                ), f'Please make sure the sum of the {choice_probs} is 1.'
        self.choice_probs = choice_probs

    @property
    def random_choice(self) -> str:
        """Sampling with probabilities."""
        assert len(self.choice_probs) == len(self._candidate_ops.keys())
        return random.choices(self.choices, weights=self.choice_probs, k=1)[0]


class OneShotPathOP(OneShotOP):
    """Design for NASBench101 search space.

    Example:
        choice = {'path': [0, 1, 2],  # a list of shape (4, )
                  'op': [0, 0, 0]}  # possible shapes: (), (1, ), (2, ), (3, )
    Args:
        candidate_ops (dict[str, dict]): the configs for the candidate
            operations.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(
        self,
        candidate_ops: nn.ModuleList,
        module_kwargs: Optional[Dict[str, Dict]] = None,
        alias: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            candidate_ops=candidate_ops,
            module_kwargs=module_kwargs,
            alias=alias,
            init_cfg=init_cfg,
        )
        assert len(candidate_ops) >= 1, (
            f'Number of candidate op must greater than 1, '
            f'but got: {len(candidate_ops)}')

        self._is_fixed = False
        self._chosen: Optional[str] = None
        self._candidate_ops = self._build_ops(candidate_ops)

    @staticmethod
    def _build_ops(candidate_ops: nn.ModuleList) -> nn.ModuleList:
        """Build candidate operations based on choice configures.

        Args:
            candidate_ops (dict[str, dict] | :obj:`nn.ModuleList`): the configs
                for the candidate operations or nn.ModuleList.
            module_kwargs (dict[str, dict], optional): Module initialization
                named arguments.

        Returns:
            ModuleList (dict[str, Any], optional):  the key of ``ops`` is
                the name of each choice in configs and the value of ``ops``
                is the corresponding candidate operation.
        """
        if isinstance(candidate_ops, nn.ModuleList):
            return candidate_ops
        else:
            raise NotImplementedError

    @property
    def random_choice(self, m=3) -> Dict:
        """choice = {'path': [0, 1, 2], 'op': [0, 0, 0]}"""
        assert m >= 1
        choice = {}
        m_ = np.random.randint(low=1, high=m + 1, size=1)[0]
        path_list = random.sample(range(m), m_)

        ops = []
        for _ in range(m_):
            ops.append(random.sample(range(3), 1)[0])
            # ops.append(random.sample(range(2), 1)[0])

        choice['op'] = ops
        choice['path'] = path_list
        return choice

    def forward_fixed(self, x: Any) -> Tensor:
        path_ids = self._chosen['path']
        op_ids = self._chosen['op']

        x_list = []
        for i, id in enumerate(path_ids):
            x_list.append(self._candidate_ops[id * 3 + op_ids[i]](x))

        x = sum(x_list)
        out = self._candidate_ops[-1](x)
        return F.relu(out)

    def forward_choice(self, x: Any, choice: Optional[Dict] = None) -> Tensor:
        import ipdb

        ipdb.set_trace()
        path_ids = choice['path']  # eg.[0, 2, 3]
        op_ids = choice['op']  # eg.[1, 1, 2]

        x_list = []
        for i, id in enumerate(path_ids):
            x_list.append(self._candidate_ops[id * 3 + op_ids[i]](x))

        x = sum(x_list)
        out = self._candidate_ops[-1](x)
        return F.relu(out)

    def fix_chosen(self, chosen: Dict) -> None:
        self.is_fixed = True
        self._chosen = chosen
