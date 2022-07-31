from typing import Dict
import time 

import torch
import torch.nn as nn
from pplib.evaluator import macro_evaluator
from pplib.evaluator.nats_evaluator import NATSEvaluator

from pplib.nas.mutators import OneShotMutator
from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer
from .registry import register_trainer
import pplib.utils.utils as utils



@register_trainer
class MacroTrainer(BaseTrainer):
    """Trainer for Macro Benchmark.

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
        mutator: OneShotMutator,
        optimizer=None,
        criterion=None,
        scheduler=None,
        device: torch.device = torch.device('cuda'),
        log_name='macro',
        searching: bool = True,
    ):
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
        
        self.evaluator = None 

    def build_evaluator(self, dataloader, bench_path, num_sample=20):
        self.evaluator = macro_evaluator(self, dataloader, bench_path, num_sample)
        
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

            # clear grad
            for p in self.model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

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

    def _forward(self, batch_inputs):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        # forward pass
        if self.searching is True:
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
        return self.model(inputs)

    def _predict(self, batch_inputs, subnet_dict: Dict = None):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        # forward pass
        if self.searching:
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
        else:
            self.mutator.set_subnet(subnet_dict)
        return self.model(inputs), labels

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
            
            if epoch % 5 == 0:
                if self.evaluator is None:
                    bench_path = './data/benchmark/benchmark_cifar10_dataset.json'
                    self.build_evaluator(val_loader, bench_path, num_sample=20)
                else:
                    kt, ps, sp = self.evaluator.compute_rank_consistency()
                    self.writer.add_scalar(
                        'kendall_tau', kt, global_step=self.current_epoch)
                    self.writer.add_scalar(
                        'pearson', ps, global_step=self.current_epoch)
                    self.writer.add_scalar(
                        'spearman', sp, global_step=self.current_epoch)

            self.writer.add_scalar(
                'train_epoch_loss', tr_loss, global_step=self.current_epoch)
            self.writer.add_scalar(
                'valid_epoch_loss', val_loss, global_step=self.current_epoch)

            self.scheduler.step()

        total_time = time.time() - total_start_time

        # final message
        self.logger.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds""")

    def metric_score(self, loader, subnet_dict: Dict = None):
        self.model.eval()

        val_loss = 0.0
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        with torch.no_grad():
            for step, batch_inputs in enumerate(loader):
                inputs, labels = batch_inputs
                inputs = self._to_device(inputs, self.device)
                labels = self._to_device(labels, self.device)

                # move to device
                outputs = self._predict(batch_inputs, subnet_dict=subnet_dict)

                # compute loss
                loss = self._compute_loss(outputs, labels)

                # compute accuracy
                n = inputs.size(0)
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                top1_vacc.update(top1.item(), n)
                top5_vacc.update(top5.item(), n)

                # accumulate loss
                val_loss += loss.item()

                # print every 20 iter
                if step % 20 == 0:
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
