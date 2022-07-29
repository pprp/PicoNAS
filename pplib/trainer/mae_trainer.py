import time

import torch
import torch.nn as nn
import torchvision

from pplib.nas.mutators import OneShotMutator
from pplib.utils.misc import convertTensor2BoardImage
from .base import BaseTrainer
from .registry import register_trainer


@register_trainer
class MAETrainer(BaseTrainer):

    def __init__(self,
                 model: nn.Module,
                 mutator: OneShotMutator,
                 criterion,
                 optimizer,
                 scheduler,
                 device=None,
                 log_name='mae',
                 searching: bool = True):
        super().__init__(
            model=model,
            mutator=mutator,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_name=log_name,
            searching=searching)

        if self.mutator is None:
            self.mutator = OneShotMutator()
            self.mutator.prepare_from_supernet(self.model)

        if self.criterion is None:
            self.criterion = nn.MSELoss()

    def _forward(self, batch_inputs):
        img, mask, _ = batch_inputs
        img = self._to_device(img, self.device)
        mask = self._to_device(mask, self.device)
        if self.searching:
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
        return self.model(img, mask)

    def _loss(self, batch_inputs) -> None:
        """Forward and compute loss. Low Level API"""
        img, mask, _ = batch_inputs
        out = self._forward(batch_inputs)
        return self._compute_loss(out, img)

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

            if i % self.print_freq == 0:
                self.logger.info(f'Step: {i} \t Train loss: {loss.item()}')
                self.writer.add_scalar(
                    'train_step_loss',
                    loss.item(),
                    global_step=i + self.current_epoch * len(loader))

        return loss

    def _validate(self, loader):
        self.model.eval()
        val_loss = 0.

        with torch.no_grad():
            for step, batch_inputs in enumerate(loader):
                # move to device
                loss = self.forward(batch_inputs, mode='loss')

                val_loss += loss.item()

            self.logger.info(f'Val loss: {val_loss / (step+1)}')
        return val_loss / (step + 1)

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

            # visualize training results
            if epoch % 5 == 0:
                batch_inputs = next(iter(train_loader))
                out = self._forward(batch_inputs)
                img_grid = torchvision.utils.make_grid(out)
                img_origin = torchvision.utils.make_grid(batch_inputs[0])

                # visualize
                self.writer.add_image(
                    'ori_img',
                    convertTensor2BoardImage(img_origin),
                    global_step=self.current_epoch,
                    dataformats='CHW')
                self.writer.add_image(
                    'mae_img',
                    convertTensor2BoardImage(img_grid),
                    global_step=self.current_epoch,
                    dataformats='CHW')

        total_time = time.time() - total_start_time

        # final message
        self.logger.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds""")
