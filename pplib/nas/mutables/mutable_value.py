import random
from typing import Any, List


class MutableValue():
    """MutableValue is a new mutable for handling search dimensions
    that are not directly related to a module, such as kernel size,
    out channels, in channels, etc
    """

    def __init__(self, candidates: List[Any], mode: str = 'max') -> None:
        super().__init__()
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

    def sample_value(self, mode: str = 'max') -> Any:
        """Get current value based on mode."""
        assert mode in self.supported_mode, \
            f'The current mode {mode} is not supported.' \
            f'Supported mode are {self.supported_mode}.'

        if mode == 'max':
            self._chosen = self._candidates[-1]
        elif mode == 'min':
            self._chosen = self._candidates[0]
        elif mode == 'random':
            self._chosen = random.sample(self._candidates, k=1)[0]

    @property
    def current_value(self):
        if self._chosen is None:
            self.sample_value(self._mode)
        return self._chosen

    @property
    def choices(self) -> Any:
        return self._candidates

    def fix_chosen(self, chosen: Any) -> None:
        """Fix mutable with choice. This function would fix the choice of
        Mutable. The :attr:`is_fixed` will be set to True and only the selected
        operations can be retained. All subclasses must implement this method.

        Note:
            This operation is irreversible.
        """
        self._is_fixed = True
        self._chosen = chosen

    @property
    def num_choices(self) -> int:
        """int: length of choices.
        """
        return len(self.choices)
