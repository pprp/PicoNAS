import torch
import torch.nn as nn

from pplib.nas.mutators import OneShotMutator
from .base import BaseTrainer


class MAETrainer(BaseTrainer):

    def __init__(self,
                 model: nn.Module,
                 mutator: OneShotMutator,
                 criterion,
                 optimizer,
                 logger_kwargs,
                 device=None):
        super().__init__(model, mutator, criterion, optimizer, logger_kwargs,
                         device)

        if self.criterion is None:
            self.criterion = nn.MSELoss()

    def _forward(self, batch_inputs):
        img, mask, _ = batch_inputs
        out = self.model(img, mask)
        return out

    def loss(self, batch_inputs) -> None:
        """Forward and compute loss. Low Level API"""
        img, mask, _ = batch_inputs
        out = self._forward(batch_inputs)
        return self._compute_loss(out, img)

    def _train(self, loader):
        self.model.train()

        for batch_inputs in loader:
            # move to device
            loss = self.forward(batch_inputs, mode='loss')

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
            loss.backward()

            # parameters update
            self.optimizer.step()

        return loss.item()

    def _validate(self, loader):
        self.model.eval()

        with torch.no_grad():
            for batch_inputs in loader:
                # move to device
                loss = self.forward(batch_inputs, mode='loss')
        return loss.item()
