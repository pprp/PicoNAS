# Copyright (c) OpenMMLab. All rights reserved.
from .diff_mutable import (DiffChoiceRoute, DiffMutable, DiffOP,
                           GumbelChoiceRoute)
from .oneshot_mutable import (OneShotChoiceRoute, OneShotMutable, OneShotOP,
                              OneShotPathOP)

__all__ = [
    'OneShotOP', 'OneShotMutable', 'DiffOP', 'DiffChoiceRoute',
    'GumbelChoiceRoute', 'DiffMutable', 'OneShotPathOP', 'OneShotChoiceRoute'
]
