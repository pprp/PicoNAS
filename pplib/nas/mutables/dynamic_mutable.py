from abc import abstractmethod
from typing import Any, Dict, Optional

from .oneshot_mutable import CHOICE_TYPE, CHOSEN_TYPE, OneShotMutable


class DynamicMutable(OneShotMutable[CHOICE_TYPE, CHOSEN_TYPE]):
    """Base class of dynamic mutables.


    Note: autoformer -> ours
        sample_parameters -> sample_parameters
        set_sample_config -> set_forward_args
        calc_sampled_param_num -> calc_sampled_params
        get_complexity -> calc_sampled_flops

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
    def calc_sampled_params(self) -> float:
        """calculate the parameter of sampled mutable"""

    @abstractmethod
    def calc_sampled_flops(self, x: Any) -> float:
        """calculate the FLOPs of sampled mutable"""

    def set_forward_args(self, choice: CHOICE_TYPE) -> None:
        """Interface for modifying the choice using partial"""
        return super().set_forward_args(choice)

    @abstractmethod
    def fix_chosen(self, chosen: CHOSEN_TYPE) -> None:
        return super().fix_chosen(chosen)

    @abstractmethod
    def sample_choice(self) -> CHOICE_TYPE:
        """sample choice on dynamic mutable"""
