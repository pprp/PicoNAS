from abc import abstractmethod
from typing import Any

from .mutable_value import MutableValue


class DynamicMixin:
    """Base class of dynamic mutables.

    Note: autoformer -> ours
        sample_parameters -> sample_parameters
        set_sample_config -> set_forward_args

    Returns:
        _type_: _description_
    """

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        assert mode in self.supported_mode
        self._mode = mode

    @property
    def is_fixed(self):
        return self._is_fixed

    @is_fixed.setter
    def is_fixed(self, is_fixed: bool):
        self._is_fixed = is_fixed

    @abstractmethod
    def sample_parameters(self, choice: Any) -> None:
        """Modify the sample property. This function would be called in
        `modify_forward` function.

        Args:
            choice (Dict): _description_
        """

    @abstractmethod
    def fix_chosen(self, chosen: Any) -> None:
        """fix chosen and remove useless operations"""

    def get_value(self, value):
        """Get value according to value type and kind.

        Args:
            value (Int / MutableValue): Input value.
            mode (str, optional): decide the return value.
                Defaults to None.
        """
        if isinstance(value, MutableValue):
            return value.current_value
        elif isinstance(value, int):
            return value
        else:
            raise f'Not support {type(value)} currently.'
