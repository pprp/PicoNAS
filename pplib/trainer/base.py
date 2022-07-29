import os
import time
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import pplib.utils.utils as utils
from pplib.utils.logging import get_logger
from pplib.utils.utils import AvgrageMeter, accuracy


class BaseTrainer:
    """Trainer

    Args:
        model ([type]): [description]
        mutator ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        scheduler ([type]): [description]
        device ([type], optional): [description]. Defaults to None.
        log_name (str, optional): [description]. Defaults to 'base'.
        searching (bool, optional): [description]. Defaults to True.
    """

    def __init__(self,
                 model,
                 mutator,
                 criterion,
                 optimizer,
                 scheduler,
                 device=None,
                 log_name='base',
                 searching: bool = True,
                 print_freq: int = 100):

        self.model = model
        self.mutator = mutator
        self.criterion = nn.CrossEntropyLoss(
        ) if criterion is None else criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = self._get_device(device) if device is None else device
        self.model.to(self.device)
        self.searching = searching
        self.log_name = log_name
        self.print_freq = print_freq

        # attributes
        self.train_loss_: List = []
        self.val_loss_: List = []
        self.current_epoch = 0

        self.logger = get_logger(self.log_name)

        writer_path = os.path.join('./logdirs', self.log_name)
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
            tr_loss, top1_tacc, top5_tacc = self._train(train_loader)

            # validate
            val_loss, top1_vacc, top5_vacc = self._validate(val_loader)

            # save ckpt
            if epoch % 10 == 0:
                utils.save_checkpoint({'state_dict': self.model.state_dict()},
                                      self.log_name,
                                      epoch + 1,
                                      tag=f'{self.log_name}_macro')

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch: {epoch + 1}/{epochs} Time: {epoch_time} Train loss: {tr_loss} Val loss: {val_loss}'  # noqa: E501
            )

            self.writer.add_scalar(
                'train_epoch_loss', tr_loss, global_step=self.current_epoch)
            self.writer.add_scalar(
                'valid_epoch_loss', val_loss, global_step=self.current_epoch)

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
            return self._loss(batch_inputs)
        elif mode == 'tensor':
            return self._forward(batch_inputs)
        elif mode == 'predict':
            return self._predict(batch_inputs)
        else:
            raise RuntimeError(f'Invalid mode: {mode}')

    def _predict(self, batch_inputs):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        return self.model(inputs)

    def _loss(self, batch_inputs) -> Tuple:
        """Forward and compute loss. Low Level API"""
        _, labels = batch_inputs
        out = self._forward(batch_inputs)
        return self._compute_loss(out, labels), out

    def _forward(self, batch_inputs) -> Tensor:
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        return self.model(inputs)

    def _train(self, loader):
        self.model.train()

        train_loss = 0.
        top1_tacc = AvgrageMeter()
        top5_tacc = AvgrageMeter()

        for step, batch_inputs in enumerate(loader):
            # get image and labels
            inputs, labels = batch_inputs
            inputs = self._to_device(inputs, self.device)
            labels = self._to_device(labels, self.device)

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # compute loss
            loss, outputs = self.forward(batch_inputs, mode='loss')

            # backprop
            loss.backward()

            # parameters update
            self.optimizer.step()

            # compute accuracy
            n = inputs.size(0)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            top1_tacc.update(top1.item(), n)
            top5_tacc.update(top5.item(), n)

            # accumulate loss
            train_loss += loss.item()

            # print every 20 iter
            if step % self.print_freq == 0:
                self.logger.info(
                    f'Step: {step} \t Train loss: {loss.item()} Top1 acc: {top1_tacc.avg} Top5 acc: {top5_tacc.avg}'
                )
                self.writer.add_scalar(
                    'train_step_loss',
                    loss.item(),
                    global_step=step + self.current_epoch * len(loader))
                self.writer.add_scalar(
                    'top1_train_acc',
                    top1_tacc.avg,
                    global_step=step + self.current_epoch * len(loader))
                self.writer.add_scalar(
                    'top5_train_acc',
                    top5_tacc.avg,
                    global_step=step + self.current_epoch * len(loader))

        return train_loss / (step + 1), top1_tacc.avg, top5_tacc.avg

    def _validate(self, loader):
        self.model.eval()

        val_loss = 0.0
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        with torch.no_grad():
            for step, batch_inputs in enumerate(loader):
                # move to device
                outputs, labels = self.forward(batch_inputs, mode='predict')

                # compute loss
                loss = self._compute_loss(outputs, labels)

                # compute accuracy
                n = labels.size(0)
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                top1_vacc.update(top1.item(), n)
                top5_vacc.update(top5.item(), n)

                # accumulate loss
                val_loss += loss.item()

                # print every 20 iter
                if step % self.print_freq == 0:
                    self.writer.add_scalar(
                        'val_step_loss',
                        loss.item(),
                        global_step=step + self.current_epoch * len(loader))
                    self.writer.add_scalar(
                        'top1_val_acc',
                        top1_vacc.avg,
                        global_step=step + self.current_epoch * len(loader))
                    self.writer.add_scalar(
                        'top5_val_acc',
                        top5_vacc.avg,
                        global_step=step + self.current_epoch * len(loader))
            self.logger.info(
                f'Val loss: {val_loss / (step + 1)} Top1 acc: {top1_vacc.avg} Top5 acc: {top5_vacc.avg}'
            )
        return val_loss / (step + 1), top1_vacc.avg, top5_vacc.avg

    def _compute_loss(self, real, target):
        real = self._to_device(real, self.device)
        target = self._to_device(target, self.device)
        return self.criterion(real, target)

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f'Device was automatically selected: {dev}'
            warnings.warn(msg)
        else:
            dev = device
        return dev

    def _to_device(self, inputs, device):
        return inputs.to(device)
