from abc import abstractmethod
from typing import Dict, Optional

from pplib.nas.mutables.base_mutable import (CHOICE_TYPE, CHOSEN_TYPE,
                                             BaseMutable)
from .mutable_value import MutableValue


class DynamicMutable(BaseMutable[CHOICE_TYPE, CHOSEN_TYPE]):
    """Base class of dynamic mutables.

    Note: autoformer -> ours
        sample_parameters -> sample_parameters
        set_sample_config -> set_forward_args

    Args:
        module_kwargs (Optional[Dict[str, Dict]], optional): _description_.
            Defaults to None.
        alias (Optional[str], optional): _description_. Defaults to None.
        init_cfg (Optional[Dict], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    def __init__(self,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)

    @abstractmethod
    def sample_parameters(self, choice: CHOICE_TYPE) -> None:
        """Modify the sample property. This function would be called in
        `modify_forward` function.

        Args:
            choice (Dict): _description_
        """

    @abstractmethod
    def fix_chosen(self, chosen: CHOSEN_TYPE) -> None:
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
