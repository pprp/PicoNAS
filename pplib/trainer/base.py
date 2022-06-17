import logging
import time
import warnings
from msilib.schema import Error

import torch
from mindspore import Tensor


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
                 logger_kwargs,
                 device=None):
        self.model = model
        self.mutator = mutator
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)

        self.model.to(self.device)

        # attributes
        self.train_loss_ = []
        self.val_loss_ = []

        logging.basicConfig(level=logging.INFO)

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
            # track epoch time
            epoch_start_time = time.time()

            # train
            tr_loss = self._train(train_loader)

            # validate
            val_loss = self._validate(val_loader)

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time
            self._logger(tr_loss, val_loss, epoch + 1, epochs, epoch_time,
                         **self.logger_kwargs)

        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds""")

    def _logger(self,
                tr_loss,
                val_loss,
                epoch,
                epochs,
                epoch_time,
                show=True,
                update_step=20):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f'Epoch {epoch}/{epochs} | Train loss: {tr_loss}'
                msg = f'{msg} | Validation loss: {val_loss}'
                msg = f'{msg} | Time/epoch: {round(epoch_time, 5)} seconds'

                logging.info(msg)

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
        try:
            loss = self.criterion(real, target)
        except Error:
            loss = self.criterion(real, target.long())
            msg = 'Target tensor has been casted from'
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        return loss

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f'Device was automatically selected: {dev}'
            warnings.warn(msg)
        else:
            dev = device

        return dev
