import time
from typing import Tuple

import torch
import torch.nn as nn
import torchvision

import pplib.utils.utils as utils
from pplib.evaluator import NATSEvaluator
from pplib.utils.misc import convertTensor2BoardImage
from .nats_trainer import NATSTrainer
from .registry import register_trainer


@register_trainer
class NATSMAETrainer(NATSTrainer):
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
        self.evaluator = None

        # for autoslim distillation
        self.distill_criterion = nn.MSELoss().to(device)

    def build_evaluator(self, dataloader, bench_path, num_sample=20):
        self.evaluator = NATSEvaluator(self, dataloader, bench_path,
                                       num_sample)

    def _forward(self, batch_inputs):
        """Network forward step. Low Level API"""
        inputs, mask, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        mask = self._to_device(mask, self.device)
        labels = self._to_device(labels, self.device)

        # forward pass
        if self.searching:
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
                loss = self._compute_loss(outputs, inputs)

                # accumulate loss
                val_loss += loss.item()

        # self.logger.info(f'Metric Score -> Val loss: {val_loss / (step + 1)}')
        return val_loss / (step + 1)

    def forward_fairnas(self, batch_inputs):
        inputs, mask, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        mask = self._to_device(mask, self.device)
        labels = self._to_device(labels, self.device)

        forward_op_lists = self.model.set_forward_cfg('fair')
        for op_list in forward_op_lists:
            output = self.model(inputs, mask, op_list)
            loss = self._compute_loss(output, inputs)
            loss.backward()
        return loss

    def forward_spos(self, batch_inputs):
        loss = self.forward(batch_inputs, mode='loss')
        loss.backward()
        return loss

    def forward_autoslim(self, batch_inputs):
        inputs, mask, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        mask = self._to_device(mask, self.device)
        labels = self._to_device(labels, self.device)

        # max supernet
        max_forward_list = self.model.set_forward_cfg('large')
        t_output = self.model(inputs, mask, max_forward_list)
        t_loss = self._compute_loss(t_output, inputs)
        t_loss.backward(retain_graph=True)

        # middle supernet
        mid_forward_lists = [
            self.model.set_forward_cfg('uni') for _ in range(2)
        ]
        for mid_forward_list in mid_forward_lists:
            output = self.model(inputs, mask, mid_forward_list)
            loss = self.distill_criterion(output, t_output)
            loss.backward(retain_graph=True)

        # min supernet
        min_forward_list = self.model.set_forward_cfg('small')
        output = self.model(inputs, mask, min_forward_list)
        loss = self.distill_criterion(output, t_output)
        loss.backward()
        return t_loss

    def _train(self, loader):
        self.model.train()

        train_loss = 0.

        for step, batch_inputs in enumerate(loader):
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # # compute loss
            # loss = self.forward(batch_inputs, mode='loss')
            # # backprop
            # loss.backward()

            # loss = self.forward_fairnas(batch_inputs)
            loss = self.forward_autoslim(batch_inputs)

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

            if epoch % 5 == 0:
                if self.evaluator is None:
                    bench_path = './data/benchmark/nats_cifar10_acc_rank.yaml'
                    self.build_evaluator(val_loader, bench_path, num_sample=50)
                else:
                    kt, ps, sp = self.evaluator.compute_rank_consistency()
                    self.writer.add_scalar(
                        'kendall_tau', kt, global_step=self.current_epoch)
                    self.writer.add_scalar(
                        'pearson', ps, global_step=self.current_epoch)
                    self.writer.add_scalar(
                        'spearman', sp, global_step=self.current_epoch)

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

            # visualize training results
            if epoch % 5 == 0:
                batch_inputs = next(iter(train_loader))
                out = self._forward(batch_inputs)

                inputs, mask, _ = batch_inputs
                img_mask = self.model.process_mask(inputs, mask)

                img_mask = torchvision.utils.make_grid(img_mask)
                img_grid = torchvision.utils.make_grid(out)
                img_origin = torchvision.utils.make_grid(batch_inputs[0])

                # visualize
                self.writer.add_image(
                    'ori_img',
                    convertTensor2BoardImage(img_origin),
                    global_step=self.current_epoch,
                    dataformats='CHW')
                self.writer.add_image(
                    'mask_img',
                    convertTensor2BoardImage(img_mask),
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
