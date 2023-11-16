from typing import Any, Dict, List, Optional

from .base_mutable import BaseMutable


class MutableValue(BaseMutable[Any, Dict]):
    """Base class for mutable value.
    A mutable value is actually a mutable that adds some functionality to a
    list containing objects of the same type.
    Args:
        value_list (list): List of value, each value must have the same type.
        default_value (any, optional): Default value, must be one in
            `value_list`. Default to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(
        self,
        value_list: List[Any],
        default_value: Optional[Any] = None,
        alias: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(alias, init_cfg)

        self._check_is_same_type(value_list)
        self._value_list = value_list

        if default_value is None:
            default_value = value_list[0]
        self.current_choice = default_value

    @staticmethod
    def _check_is_same_type(value_list: List[Any]) -> None:
        """Check whether value in `value_list` has the same type."""
        if len(value_list) == 1:
            return

        for i in range(1, len(value_list)):
            is_same_type = type(value_list[i - 1]) is type(value_list[i])  # noqa: E721
            if not is_same_type:
                raise TypeError(
                    'All elements in `value_list` must have same '
                    f'type, but both types {type(value_list[i-1])} '
                    f'and type {type(value_list[i])} exist.'
                )

    @property
    def choices(self) -> List[Any]:
        """List of choices."""
        return self._value_list

    def fix_chosen(self, chosen: Dict[str, Any]) -> None:
        """Fix mutable value with subnet config.
        Args:
            chosen (dict): the information of chosen.
        """
        if self.is_fixed:
            raise RuntimeError('MutableValue can not be fixed twice')

        all_choices = chosen['all_choices']
        current_choice = chosen['current_choice']

        assert (
            all_choices == self.choices
        ), f'Expect choices to be: {self.choices}, but got: {all_choices}'
        assert current_choice in self.choices

        self.current_choice = current_choice
        self.is_fixed = True

    def dump_chosen(self) -> Dict[str, Any]:
        """Dump information of chosen.
        Returns:
            Dict[str, Any]: Dumped information.
        """
        return dict(current_choice=self.current_choice, all_choices=self.choices)

    @property
    def num_choices(self) -> int:
        """Number of all choices.
        Returns:
            int: Number of choices.
        """
        return len(self.choices)

    @property
    def current_choice(self) -> Optional[Any]:
        """Current choice of mutable value."""
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: Any) -> Any:
        """Setter of current choice."""
        if choice not in self.choices:
            raise ValueError(
                f'Expected choice in: {self.choices}, ' f'but got: {choice}'
            )

        self._current_choice = choice

    def __rmul__(self, other):
        """Please refer to method :func:`__mul__`."""
        return self * other

    def __mul__(self, other: int):
        """Overload `*` operator.
        Args:
            other (int): Expand ratio.
        Returns:
            DerivedMutable: Derived expand mutable.
        """
        if isinstance(other, int):
            return self.current_choice * other
        raise TypeError(f'Unsupported type {type(other)} for mul!')

    def __floordiv__(self, other: int):
        """Overload `//` operator.
        Args:
            other: (int, tuple): divide ratio for int or
                (divide ratio, divisor) for tuple.
        Returns:
            DerivedMutable: Derived divide mutable.
        """
        if isinstance(other, int):
            return self.current_choice // other
        raise TypeError(f'Unsupported type {type(other)} for div!')

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(value_list={self._value_list}, '
        s += f'current_choice={self.current_choice})'
        return s
