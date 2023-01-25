from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from nanonas.nas.mutators import OneShotMutator
from .base import BaseTrainer
from .registry import register_trainer


@register_trainer
class SPOSTrainer(BaseTrainer):

    def __init__(
        self,
        model: nn.Module,
        mutator: OneShotMutator,
        criterion,
        optimizer,
        scheduler,
        device: torch.device = None,
        log_name: str = 'spos',
        searching: bool = True,
        print_freq: int = 100,
        dataset: str = 'cifar10',
        **kwargs,
    ):
        super().__init__(
            model=model,
            mutator=mutator,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_name=log_name,
            searching=searching,
            print_freq=print_freq,
            dataset=dataset,
            **kwargs)
        if self.mutator is None:
            self.mutator = OneShotMutator()
            self.mutator.prepare_from_supernet(model)

    def _forward(self, batch_inputs) -> Tensor:
        """Network forward step. Low Level API"""
        inputs, _ = batch_inputs
        inputs = self._to_device(inputs, self.device)

        if self.searching:
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
        return self.model(inputs)

    def _predict(self, batch_inputs, subnet_dict: Dict = None):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        if self.searching:
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
        else:
            self.mutator.set_subnet(subnet_dict)
        return self.model(inputs), labels
