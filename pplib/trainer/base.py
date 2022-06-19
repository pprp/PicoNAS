import os
import time
import warnings

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from pplib.utils.logging import get_logger


class BaseTrainer:
    """Trainer

    Class that eases the training of a PyTorch model.

    Args:
        model : torch.Module
            The model to train.
        criterion : torch.Module
            Loss function criterion.
        optimizer : torch.optim
            Optimizer to perform the parameters update.
        logger_kwards : dict
            Args for ..
    """

    def __init__(self,
                 model,
                 mutator,
                 criterion,
                 optimizer,
                 scheduler,
                 device=None,
                 log_name='base',
                 searching: bool = True):
        self.model = model
        self.mutator = mutator
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.searching = searching

        # attributes
        self.train_loss_ = []
        self.val_loss_ = []
        self.current_epoch = 0

        self.logger = get_logger(log_name)

        writer_path = os.path.join('./logdirs', log_name)
        self.writer = SummaryWriter(writer_path)

    def fit(self, train_loader, val_loader, epochs):
        """Fits. High Level API

        Fit the model using the given loaders for the given number
        of epochs.

        Args:
            train_loader :
            val_loader :
            epochs : int
                Number of training epochs.

        """
        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        for epoch in range(epochs):
            self.current_epoch = epoch
            # track epoch time
            epoch_start_time = time.time()

            # train
            tr_loss = self._train(train_loader)

            # validate
            val_loss = self._validate(val_loader)

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch: {epoch + 1}/{epochs} Time: {epoch_time} Train loss: {tr_loss.item()} Val loss: {val_loss.item()}'  # noqa: E501
            )

            self.writer.add_scalar(
                'train_epoch_loss',
                tr_loss.item(),
                global_step=self.current_epoch)
            self.writer.add_scalar(
                'valid_epoch_loss',
                val_loss.item(),
                global_step=self.current_epoch)

            self.scheduler.step()

        total_time = time.time() - total_start_time

        # final message
        self.logger.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds""")

    def forward(self,
                batch_inputs: torch.Tensor,
                mode: str = 'tensor') -> Tensor:
        """Forward. High Level API.

        Note:
            if model == 'loss', return dict of loss tensor;
            if model == 'tensor', return naive tensor type results;
            if model == 'predict', called by val_step and test_step results.

        Args:
            batch_inputs (torch.Tensor): _description_
            mode (str, optional): _description_. Defaults to 'tensor'.
        """
        if mode == 'loss':
            return self.loss(batch_inputs)
        elif mode == 'tensor':
            return self._forward(batch_inputs)
        elif mode == 'predict':
            return self.predict(batch_inputs)
        else:
            raise RuntimeError(f'Invalid mode: {mode}')

    def predict(self, batch_inputs):
        """Network forward step. Low Level API"""
        features, labels = batch_inputs
        features, labels = self._to_device(features, labels, self.device)
        # forward pass
        out = self.model(features)
        return out

    def loss(self, batch_inputs) -> Tensor:
        """Forward and compute loss. Low Level API"""
        _, labels = batch_inputs
        out = self._forward(batch_inputs)
        return self._compute_loss(out, labels)

    def _forward(self, batch_inputs) -> Tensor:
        """Network forward step. Low Level API"""
        features, labels = batch_inputs
        features, labels = self._to_device(features, labels, self.device)
        # forward pass
        out = self.model(features)
        return out

    def _train(self, loader):
        self.model.train()

        for i, batch_inputs in enumerate(loader):
            # move to device
            loss = self.forward(batch_inputs, mode='loss')

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
            loss.backward()

            # parameters update
            self.optimizer.step()

            if i % 20 == 0:
                self.logger.info(f'Step: {i} \t Train loss: {loss.item()}')
                self.writer.add_scalar(
                    'train_step_loss',
                    loss.item(),
                    global_step=i + self.current_epoch * len(loader))

        return loss.item()

    def _to_device(self, features, labels, device):
        return features.to(device), labels.to(device)

    def _validate(self, loader):
        self.model.eval()

        with torch.no_grad():
            for batch_inputs in loader:
                # move to device
                loss = self.forward(batch_inputs, mode='loss')
        return loss.item()

    def _compute_loss(self, real, target):
        # print(real.shape, target.shape)
        real, target = self._to_device(real, target, self.device)
        loss = self.criterion(real, target)
        return loss

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f'Device was automatically selected: {dev}'
            warnings.warn(msg)
        else:
            dev = device

        return dev
