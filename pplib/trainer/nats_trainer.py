import time
from typing import Tuple

import torch
import torch.nn as nn

import pplib.utils.utils as utils
from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer


class NATSTrainer(BaseTrainer):
    """Trainer for NATS-Bench.

    Args:
        model (nn.Module): _description_
        dataloader (Dict): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        scheduler (_type_): _description_
        epochs (int): _description_
        searching (bool, optional): _description_. Defaults to True.
        num_choices (int, optional): _description_. Defaults to 4.
        num_layers (int, optional): _description_. Defaults to 20.
        device (torch.device, optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        mutator,
        optimizer=None,
        criterion=None,
        scheduler=None,
        device: torch.device = torch.device('cuda'),
        log_name='nats',
        searching: bool = True,
        method: str = 'uni',
    ):
        super().__init__(model, mutator, criterion, optimizer, scheduler,
                         device, log_name, searching)

        assert method in {'uni', 'fair'}
        self.method = method

    def _loss(self, batch_inputs) -> Tuple:
        """Forward and compute loss. Low Level API"""
        inputs, labels = batch_inputs
        out = self._forward(batch_inputs)
        return self._compute_loss(out, labels), out, labels, inputs.size(0)

    def _train(self, loader):
        self.model.train()

        train_loss = 0.
        top1_tacc = AvgrageMeter()
        top5_tacc = AvgrageMeter()

        for step, batch_inputs in enumerate(loader):
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # compute loss
            loss, outputs, labels = self.forward(batch_inputs, mode='loss')

            # backprop
            loss.backward()

            # clear grad
            for p in self.model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

            # parameters update
            self.optimizer.step()

            # compute accuracy
            n = labels.size(0)
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

    def _forward(self, batch_inputs):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # forward pass
        if self.searching is True:
            # rand_subnet = self.mutator.random_subnet
            # self.mutator.set_subnet(rand_subnet)
            forward_op_list = self.model.set_forward_cfg(self.method)
        return self.model(inputs, list(forward_op_list))

    def _predict(self, batch_inputs, current_op_list):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        # forward pass
        if self.searching:
            forward_op_list = self.model.set_forward_cfg(self.method)
        else:
            forward_op_list = current_op_list
        return self.model(inputs, forward_op_list), labels

    def metric_score(self, loader, current_op_list):
        self.model.eval()

        val_loss = 0.0
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        with torch.no_grad():
            for step, batch_inputs in enumerate(loader):
                # move to device
                outputs, labels = self._predict(batch_inputs, current_op_list)

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
                if step % 50 == 0:
                    self.logger.info(
                        f'Step: {step} \t Val loss: {loss.item()} Top1 acc: {top1_vacc.avg} Top5 acc: {top5_vacc.avg}'
                    )
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

        return val_loss / (step + 1), top1_vacc.avg, top5_vacc.avg


class MAENATSTrainer(NATSTrainer):
    """Main difference rely on the forward function."""

    def __init__(
        self,
        model: nn.Module,
        mutator,
        optimizer=None,
        criterion=None,
        scheduler=None,
        device: torch.device = torch.device('cuda'),
        log_name='nats',
        searching: bool = True,
        method: str = 'uni',
    ):
        super().__init__(model, mutator, optimizer, criterion, scheduler,
                         device, log_name, searching, method)

        assert method in {'uni', 'fair'}
        self.method = method

    def _forward(self, batch_inputs):
        """Network forward step. Low Level API"""
        inputs, mask, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        mask = self._to_device(mask, self.device)
        labels = self._to_device(labels, self.device)

        # forward pass
        if self.searching is True:
            # rand_subnet = self.mutator.random_subnet
            # self.mutator.set_subnet(rand_subnet)
            forward_op_list = self.model.set_forward_cfg(self.method)
        return self.model(inputs, mask, list(forward_op_list))

    def _predict(self, batch_inputs, current_op_list=None):
        """Network forward step. Low Level API"""
        inputs, mask, _ = batch_inputs
        inputs = self._to_device(inputs, self.device)
        mask = self._to_device(mask, self.device)

        # forward pass
        if self.searching:
            forward_op_list = self.model.set_forward_cfg(self.method)
        else:
            forward_op_list = current_op_list if current_op_list is not None else self.model.set_forward_cfg(
                self.method)

        return self.model(inputs, mask, forward_op_list), inputs

    def _loss(self, batch_inputs) -> Tuple:
        """Forward and compute loss. Low Level API"""
        inputs, _, _ = batch_inputs
        out = self._forward(batch_inputs)
        return self._compute_loss(out, inputs)

    def metric_score(self, loader, current_op_list):
        self.model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for step, batch_inputs in enumerate(loader):
                # move to device
                outputs, inputs = self._predict(batch_inputs, current_op_list)

                # compute loss
                # import ipdb; ipdb.set_trace()
                loss = self._compute_loss(outputs, inputs)

                # accumulate loss
                val_loss += loss.item()

                # print every 20 iter
                if step % 30 == 0:
                    self.logger.info(
                        f'Step: {step} \t Val loss: {loss.item()}')

        return val_loss / (step + 1)

    def _train(self, loader):
        self.model.train()

        train_loss = 0.

        for step, batch_inputs in enumerate(loader):
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # compute loss
            loss = self.forward(batch_inputs, mode='loss')

            # backprop
            loss.backward()

            # clear grad
            for p in self.model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

            # parameters update
            self.optimizer.step()

            # accumulate loss
            train_loss += loss.item()

            # print every 20 iter
            if step % self.print_freq == 0:
                self.logger.info(f'Step: {step} \t Train loss: {loss.item()}')
                self.writer.add_scalar(
                    'train_step_loss',
                    loss.item(),
                    global_step=step + self.current_epoch * len(loader))

        return train_loss / (step + 1)

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

    def _validate(self, loader):
        self.model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for step, batch_inputs in enumerate(loader):
                # move to device
                outputs, inputs = self.forward(batch_inputs, mode='predict')

                # compute loss
                loss = self._compute_loss(outputs, inputs)

                # accumulate loss
                val_loss += loss.item()

                # print every 20 iter
                if step % self.print_freq == 0:
                    self.logger.info(
                        f'Step: {step} \t Val loss: {loss.item()}')
                    self.writer.add_scalar(
                        'val_step_loss',
                        loss.item(),
                        global_step=step + self.current_epoch * len(loader))

        return val_loss / (step + 1)
