import random
from typing import Any, List, Optional

from pplib.nas.mutables.base_mutable import (CHOICE_TYPE, CHOSEN_TYPE,
                                             BaseMutable)


class MutableValue(BaseMutable[CHOICE_TYPE, CHOSEN_TYPE]):
    """MutableValue is a new mutable for handling search dimensions
    that are not directly related to a module, such as kernel size,
    out channels, in channels, etc
    """

    def __init__(self,
                 candidates: List[Any],
                 mode: str = 'max',
                 alias: Optional[str] = None) -> None:
        super().__init__(alias=alias)
        self.supported_mode = ['random', 'min', 'max']
        assert mode in self.supported_mode, \
            f'The current mode {mode} is not supported.' \
            f'Supported mode are {self.supported_mode}.'

        self._candidates = candidates
        self._chosen = None
        self._mode = 'max'

        # sort the candidates
        self._sort_candidates()

    def _sort_candidates(self) -> None:
        """Sort candidates list"""
        self._candidates = sorted(self._candidates)

    @property
    def current_value(self, mode: str = 'max') -> Any:
        """Get current value based on mode."""
        if self._chosen is not None:
            return self._chosen

        if mode == 'max':
            return self._candidates[-1]
        elif mode == 'min':
            return self._candidates[0]
        elif mode == 'random':
            return random.sample(self._candidates, k=1)[0]

    @property
    def choices(self) -> List[CHOSEN_TYPE]:
        return self._candidates

    def fix_chosen(self, chosen: CHOICE_TYPE) -> None:
        """Fix mutable with choice. This function would fix the choice of
        Mutable. The :attr:`is_fixed` will be set to True and only the selected
        operations can be retained. All subclasses must implement this method.

        Note:
            This operation is irreversible.
        """
        self._chosen = chosen
