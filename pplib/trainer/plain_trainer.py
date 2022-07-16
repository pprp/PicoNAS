import torch.nn as nn

from pplib.nas.mutators import OneShotMutator
from .base import BaseTrainer


class PlainTrainer(BaseTrainer):
    """Used for train fixed subnet.

    Args:
        BaseTrainer (_type_): _description_
    """

    def __init__(self,
                 model: nn.Module,
                 mutator: OneShotMutator,
                 criterion,
                 optimizer,
                 scheduler,
                 device=None,
                 log_name='plain',
                 searching: bool = True):
        super().__init__(model, mutator, criterion, optimizer, scheduler,
                         device, log_name, searching)
