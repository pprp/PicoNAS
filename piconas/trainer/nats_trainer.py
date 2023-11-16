import time
from typing import Tuple

import torch
import torch.nn as nn

import piconas.utils.utils as utils
from piconas.evaluator import NATSEvaluator
from piconas.utils.utils import AvgrageMeter, accuracy
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
        dataset: str = 'dataset',
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
            dataset=dataset,
            **kwargs,
        )

        assert method in {'uni', 'fair'}
        self.method = method
        self.evaluator = None

    def build_evaluator(self, num_sample=50):
        self.evaluator = NATSEvaluator(self, num_sample=num_sample)

    def _loss(self, batch_inputs) -> Tuple:
        """Forward and compute loss. Low Level API"""
        inputs, labels = batch_inputs
        labels = self._to_device(labels, self.device)
        out = self._forward(batch_inputs)
        return self._compute_loss(out, labels), out

    def _train(self, loader):
        self.model.train()

        train_loss = 0.0
        top1_tacc = AvgrageMeter()
        top5_tacc = AvgrageMeter()

        for step, batch_inputs in enumerate(loader):
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            labels = self._to_device(batch_inputs[1], self.device)

            # compute loss
            # loss, outputs = self.forward_spos(batch_inputs)
            loss, outputs = self.forward_fairnas(batch_inputs)

            # clear grad
            for p in self.model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

            # parameters update
            self.optimizer.step()

            # compute accuracy
            n = outputs.size(0)
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
                    'STEP_LOSS/train_step_loss',
                    loss.item(),
                    global_step=step + self.current_epoch * len(loader),
                )
                self.writer.add_scalar(
                    'TRAIN_ACC/top1_train_acc',
                    top1_tacc.avg,
                    global_step=step + self.current_epoch * len(loader),
                )
                self.writer.add_scalar(
                    'TRAIN_ACC/top5_train_acc',
                    top5_tacc.avg,
                    global_step=step + self.current_epoch * len(loader),
                )

        return train_loss / (step + 1), top1_tacc.avg, top5_tacc.avg

    def _forward(self, batch_inputs):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # forward pass
        if self.searching is True:
            forward_op_list = self.model.set_forward_cfg(self.method)
        return self.model(inputs, list(forward_op_list))

    def _predict(self, batch_inputs, current_op_list=None):
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
        # self.model.eval()

        val_loss = 0.0
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        with torch.no_grad():
            for batch_inputs in loader:
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

        # val_loss / (step + 1), top1_vacc.avg, top5_vacc.avg
        return top1_vacc.avg

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

            if epoch % 5 == 0:
                if self.evaluator is None:
                    self.build_evaluator(val_loader, num_sample=50)
                else:
                    kt, ps, sp = self.evaluator.compute_rank_consistency()
                    self.writer.add_scalar(
                        'RANK/kendall_tau', kt, global_step=self.current_epoch
                    )
                    self.writer.add_scalar(
                        'RANK/pearson', ps, global_step=self.current_epoch
                    )
                    self.writer.add_scalar(
                        'RANK/spearman', sp, global_step=self.current_epoch
                    )

            # save ckpt
            if epoch % 10 == 0:
                utils.save_checkpoint(
                    {'state_dict': self.model.state_dict()},
                    self.log_name,
                    epoch + 1,
                    tag=f'{self.log_name}_macro',
                )

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch: {epoch + 1}/{epochs} Time: {epoch_time} Train loss: {tr_loss} Val loss: {val_loss}'  # noqa: E501
            )

            self.writer.add_scalar(
                'EPOCH_LOSS/train_epoch_loss', tr_loss, global_step=self.current_epoch
            )
            self.writer.add_scalar(
                'EPOCH_LOSS/valid_epoch_loss', val_loss, global_step=self.current_epoch
            )

            self.scheduler.step()

        total_time = time.time() - total_start_time

        # final message
        self.logger.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )

    def forward_fairnas(self, batch_inputs):
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        loss_list = []

        forward_op_lists = self.model.set_forward_cfg('fair')
        for op_list in forward_op_lists:
            output = self.model(inputs, op_list)
            loss = self._compute_loss(output, labels)
            loss_list.append(loss)
        sum_loss = sum(loss_list)
        sum_loss.backward()
        return sum_loss, output

    def forward_spos(self, batch_inputs):
        loss, outputs = self.forward(batch_inputs, mode='loss')
        loss.backward()
        return loss, outputs

    def forward_autoslim(self, batch_inputs):
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # max supernet
        max_forward_list = self.model.set_forward_cfg('large')
        t_output, t_feat = self.model(inputs, max_forward_list)
        t_loss = self._compute_loss(t_output, labels)
        t_loss.backward(retain_graph=True)

        # middle supernet
        mid_forward_lists = [
            self.model.set_forward_cfg('uni') for _ in range(2)]
        for mid_forward_list in mid_forward_lists:
            output, s_feat = self.model(inputs, mid_forward_list)
            loss = self.distill_criterion(output, t_output)
            loss.backward(retain_graph=True)

        # min supernet
        min_forward_list = self.model.set_forward_cfg('small')
        output, s_feat = self.model(inputs, min_forward_list)
        loss = self.distill_criterion(output, t_output)
        loss.backward()
        return t_loss, output

    def forward_cc_autoslim(self, batch_inputs):
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        mse_loss_list = []
        cc_loss_list = []

        # max supernet
        max_forward_list = self.model.set_forward_cfg('large')
        t_output, feat_t = self.model(inputs, max_forward_list)
        t_loss = self._compute_loss(t_output, labels)
        mse_loss_list.append(t_loss)

        # middle supernet
        mid_forward_lists = [
            self.model.set_forward_cfg('uni') for _ in range(2)]
        for mid_forward_list in mid_forward_lists:
            output, feat_s = self.model(inputs, mid_forward_list)
            loss = self.distill_criterion(output, t_output)
            cc_loss = self.cc_distill(feat_s, feat_t) * self.lambda_kd

            mse_loss_list.append(loss)
            cc_loss_list.append(cc_loss)

        # min supernet
        min_forward_list = self.model.set_forward_cfg('small')
        output, feat_s = self.model(inputs, min_forward_list)
        loss = self.distill_criterion(output, t_output)
        cc_loss = self.cc_distill(feat_s, feat_t) * self.lambda_kd

        mse_loss_list.append(loss)
        cc_loss_list.append(cc_loss)

        sum_loss = sum(mse_loss_list) + sum(cc_loss_list) * self.lambda_kd
        sum_loss.backward()

        # self.logger.info(f"mse loss: {sum(mse_loss_list).item()} cc loss: {sum(cc_loss_list).item()}")

        return t_loss, output

    def _validate(self, loader):
        # self.model.eval()

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
                        'STEP_LOSS/valid_step_loss',
                        loss.item(),
                        global_step=step + self.current_epoch * len(loader),
                    )
                    self.writer.add_scalar(
                        'VAL_ACC/top1_val_acc',
                        top1_vacc.avg,
                        global_step=step + self.current_epoch * len(loader),
                    )
                    self.writer.add_scalar(
                        'VAL_ACC/top5_val_acc',
                        top5_vacc.avg,
                        global_step=step + self.current_epoch * len(loader),
                    )
            self.logger.info(
                f'Val loss: {val_loss / (step + 1)} Top1 acc: {top1_vacc.avg} Top5 acc: {top5_vacc.avg}'
            )
        return val_loss / (step + 1), top1_vacc.avg, top5_vacc.avg
