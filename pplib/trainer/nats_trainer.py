from typing import Tuple

import torch
import torch.nn as nn

from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer
from .registry import register_trainer


@register_trainer
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
