import copy
from typing import List

import torch
import torch.nn as nn

from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer


class Distill_Trainer(BaseTrainer):
    """Trainer for distill from resnet56 to resnet20.

    Args:
        model (List): First is Teacher network, second is
            student network.
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
        model: List,
        mutator=None,
        optimizer=None,
        criterion=None,
        scheduler=None,
        device: torch.device = torch.device('cuda'),
        log_name='nasbench201',
        searching: bool = True,
        dataset: str = 'cifar10',
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

        self.num_choices = 2
        self.learnable_params = nn.Parameter(torch.randn(self.num_choices) * 2)

        self.arch_optimizer = torch.optim.Adam(
            self.learnable_params.parameters(),
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3,
        )

        # unroll choice
        self.unroll = False

    def _train(self, train_loader, valid_loader):
        self.model.train()
        train_loss = 0.0
        top1_tacc = AvgrageMeter()
        top5_tacc = AvgrageMeter()

        for step, ((trn_x, trn_y),
                   (val_x,
                    val_y)) in enumerate(zip(train_loader, valid_loader)):
            # get image and labels
            trn_x = self._to_device(trn_x, self.device)
            trn_y = self._to_device(trn_y, self.device)
            val_x = self._to_device(val_x, self.device)
            val_y = self._to_device(val_y, self.device)

            # phase 1: arch parameter update
            self.arch_optimizer.zero_grad()
            if self.unroll:
                self._unrolled_backward(trn_x, trn_y, val_x, val_y)
            else:
                output = self.model(val_x)
                loss = self.criterion(output, val_y)
                loss.backward()
            self.arch_optimizer.step()

            # phase 2: supernet parameter update
            self.optimizer.zero_grad()
            outputs = self.model(trn_x)
            loss = self.criterion(outputs, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()

            # compute accuracy
            n = trn_x.size(0)
            top1, top5 = accuracy(outputs, trn_y, topk=(1, 5))
            top1_tacc.update(top1.item(), n)
            top5_tacc.update(top5.item(), n)

            # accumulate loss
            train_loss += loss.item()

            # print every 20 iter
            if step % self.print_freq == 0:
                self.logger.info(
                    f'Step: {step:03} Train loss: {loss.item():.4f} Top1 acc: {top1_tacc.avg:.3f} Top5 acc: {top5_tacc.avg:.3f}'
                )
                self.writer.add_scalar(
                    'STEP_LOSS/train_step_loss',
                    loss.item(),
                    global_step=step + self.current_epoch * len(train_loader),
                )
                self.writer.add_scalar(
                    'TRAIN_ACC/top1_train_acc',
                    top1_tacc.avg,
                    global_step=step + self.current_epoch * len(train_loader),
                )
                self.writer.add_scalar(
                    'TRAIN_ACC/top5_train_acc',
                    top5_tacc.avg,
                    global_step=step + self.current_epoch * len(train_loader),
                )

        # FOR DEBUG
        for k, v in self.learnable_params.items():
            self.logger.info(
                f'current arch_param: key: {k}: value: {nn.functional.softmax(v, dim=-1).cpu()}'
            )

        return train_loss / (step + 1), top1_tacc.avg, top5_tacc.avg

    def _unrolled_backward(self, trn_x, trn_y, val_x, val_y):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        # do virtual step on training data
        lr = self.optimizer.param_groups[0]['lr']
        momentum = self.optimizer.param_groups[0]['momentum']
        weight_decay = self.optimizer.param_groups[0]['weight_decay']
        self._compute_virtual_model(trn_x, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        output = self.model(val_x)
        loss = self.criterion(output, val_y)

        w_model, w_arch = tuple(self.model.parameters()), tuple(
            self.mutator.parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_arch)
        d_model, d_arch = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients [expression (8) from paper]
        hessian = self._compute_hessian(backup_params, d_model, trn_x, trn_y)
        with torch.no_grad():
            # Compute expression (7) from paper
            for param, d, h in zip(w_arch, d_arch, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, x, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        output = self.model(x)
        loss = self.criterion(output, y)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.optimizer.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, trn_x, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1e-08:
            self.logger.warning(
                f'In computing hessian, norm is smaller than 1E-8, cause eps to be {norm.item()}.'
            )
        dalphas = []
        for e in [eps, -2. * eps]:
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d
            output = self.model(trn_x)
            loss = self.criterion(output, trn_y)
            dalphas.append(
                torch.autograd.grad(loss, self.mutator.parameters()))
        dalpha_pos, dalpha_neg = dalphas
        return [(p - n) / 2.0 * eps for p, n in zip(dalpha_pos, dalpha_neg)]
