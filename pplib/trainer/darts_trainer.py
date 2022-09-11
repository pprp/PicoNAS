import copy
import time
from typing import Dict

import torch
import torch.nn as nn
from mmcv.cnn import get_model_complexity_info

import pplib.utils.utils as utils
from pplib.core.losses import PairwiseRankLoss
# from pplib.evaluator.nb201_evaluator import NB201Evaluator
from pplib.models.nasbench201 import DiffNASBench201Network
from pplib.nas.mutators import DiffMutator
from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer
from .registry import register_trainer


@register_trainer
class Darts_Trainer(BaseTrainer):
    """Trainer for NB201 Benchmark with Darts algo.

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
        model: DiffNASBench201Network,
        mutator: DiffMutator,
        optimizer=None,
        criterion=None,
        scheduler=None,
        device: torch.device = torch.device('cuda'),
        log_name='nasbench201',
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
            searching=searching,
        )

        # init flops
        self._init_flops()

        if self.mutator is None:
            # Note: use alias to build search group
            self.mutator = DiffMutator(with_alias=True)
            self.mutator.prepare_from_supernet(model)
            self.mutator.arch_params.to(self.device)

        # evaluate the rank consistency
        # self.evaluator = self._build_evaluator(num_sample=50)

        # pairwise rank loss
        self.pairwise_rankloss = PairwiseRankLoss()

        # record current rand_subnet
        self.rand_subnet = None

        # optimizer for arch; origin optimizer for supernet
        self.arch_optimizer = torch.optim.Adam(
            self.mutator.parameters(),
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3,
        )

        # unroll
        self.unroll = False

    # def _build_evaluator(self, num_sample=50):
    #     return NB201Evaluator(self, num_sample)

    def _train(self, train_loader, valid_loader):
        train_loss = 0.0
        top1_tacc = AvgrageMeter()
        top5_tacc = AvgrageMeter()

        for step, (train_batch,
                   valid_batch) in enumerate(zip(train_loader, valid_loader)):
            # get image and labels
            trn_x, trn_y = train_batch
            val_x, val_y = valid_batch

            trn_x = self._to_device(trn_x, self.device)
            trn_y = self._to_device(trn_y, self.device)
            val_x = self._to_device(val_x, self.device)
            val_y = self._to_device(val_y, self.device)

            # phase 1: arch parameter update
            # self.model.eval()
            self.mutator.train()
            self.arch_optimizer.zero_grad()
            if self.unroll:
                self._unrolled_backward(trn_x, trn_y, val_x, val_y)
            else:
                output = self.model(val_x)
                loss = self.criterion(output, val_y)
                loss.backward()
            self.arch_optimizer.step()

            # phase 2: supernet parameter update
            self.model.train()
            # self.mutator.eval()
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
        for k, v in self.mutator.arch_params.items():
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
        for e in [eps, -2.0 * eps]:
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d
            output = self.model(trn_x)
            loss = self.criterion(output, trn_y)
            dalphas.append(
                torch.autograd.grad(loss, self.mutator.parameters()))
        dalpha_pos, dalpha_neg = dalphas
        return [(p - n) / 2.0 * eps for p, n in zip(dalpha_pos, dalpha_neg)]

    def search_subnet(self):
        """Search subnet by mutator."""
        return self.mutator.sample_choices()

    def _predict(self, batch_inputs, subnet_dict: Dict = None):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # forward pass
        if subnet_dict is not None:
            self.mutator.set_subnet(subnet_dict)
        return self.model(inputs), labels

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
                f'Val loss: {val_loss / (step + 1)} Top1 acc: {top1_vacc.avg}'
                f' Top5 acc: {top5_vacc.avg}')
        return val_loss / (step + 1), top1_vacc.avg, top5_vacc.avg

    def fit(self, train_loader, val_loader, epochs):
        """Fits. High Level API
        Fit the model using the given loaders for the given number
        of epochs.
        """
        # track total training time
        total_start_time = time.time()

        # record max epoch
        self.max_epochs = epochs

        # ---- train process ----
        for epoch in range(epochs):
            self.current_epoch = epoch
            # track epoch time
            epoch_start_time = time.time()

            # train
            tr_loss, top1_tacc, top5_tacc = self._train(
                train_loader, val_loader)

            # validate
            val_loss, top1_vacc, top5_vacc = self._validate(val_loader)

            # save ckpt
            if epoch % 10 == 0:
                utils.save_checkpoint(
                    {'state_dict': self.model.state_dict()},
                    self.log_name,
                    epoch + 1,
                    tag=f'{self.log_name}_nb201',
                )

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch: {epoch + 1}/{epochs} Time: {epoch_time} Train loss: {tr_loss} Val loss: {val_loss}'  # noqa: E501
            )

            self.logger.info(f'==> export subnet: {self.search_subnet()}')

            # if epoch % 5 == 0:
            #     assert self.evaluator is not None
            #     kt, ps, sp = self.evaluator.compute_rank_consistency(
            #         val_loader, self.mutator)
            #     self.writer.add_scalar(
            #         'RANK/kendall_tau', kt, global_step=self.current_epoch)
            #     self.writer.add_scalar(
            #         'RANK/pearson', ps, global_step=self.current_epoch)
            #     self.writer.add_scalar(
            #         'RANK/spearman', sp, global_step=self.current_epoch)

            self.writer.add_scalar(
                'EPOCH_LOSS/train_epoch_loss',
                tr_loss,
                global_step=self.current_epoch)
            self.writer.add_scalar(
                'EPOCH_LOSS/valid_epoch_loss',
                val_loss,
                global_step=self.current_epoch)

            self.scheduler.step()

        total_time = time.time() - total_start_time

        # final message
        self.logger.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds""")

    def metric_score(self, loader, subnet_dict: Dict = None):
        # self.model.eval()

        val_loss = 0.0
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        with torch.no_grad():
            for step, batch_inputs in enumerate(loader):
                # move to device
                outputs, labels = self._predict(
                    batch_inputs, subnet_dict=subnet_dict)

                # compute loss
                loss = self._compute_loss(outputs, labels)

                # compute accuracy
                n = labels.size(0)
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                top1_vacc.update(top1.item(), n)
                top5_vacc.update(top5.item(), n)

                # accumulate loss
                val_loss += loss.item()

                # print every 50 iter
                if step % 50 == 0:
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
            # self.logger.info(
            #     f'Val loss: {loss.item()}'
            #     f'Top1 acc: {top1_vacc.avg} Top5 acc: {top5_vacc.avg}')

        return top1_vacc.avg

    def _init_flops(self):
        """generate flops."""
        self.model.eval()
        # Note 1: after this process, each module in self.model
        #       would have the __flops__ attribute.
        # Note 2: this function should be called before
        #       mutator.prepare_from_supernet()
        flops, params = get_model_complexity_info(self.model, self.input_shape)
        self.model.train()
        return flops, params

    def get_subnet_flops(self, subnet_dict) -> float:
        """Calculate current subnet flops based on config."""
        subnet_flops = 0
        for k, v in self.mutator.search_group.items():
            current_choice = subnet_dict[k]  # '1' or '2' or 'I'
            choice_flops = 0
            for _, module in v[0]._candidate_ops[current_choice].named_modules(
            ):
                flops = getattr(module, '__flops__', 0)
                if flops > 0:
                    choice_flops += flops
            # print(f'k: {k} choice: {current_choice} flops: {choice_flops}')
            subnet_flops += choice_flops
        return subnet_flops

    def get_subnet_error(self,
                         subnet_dict: Dict,
                         train_loader=None,
                         val_loader=None) -> float:
        """Calculate the subnet of validation error.
        Including:
        1. BN calibration
        2. Start test
        """
        # Process dataloader
        assert train_loader is not None
        assert val_loader is not None

        # Info about dataloader
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        max_train_iters = 200
        max_test_iters = 40

        self.mutator.set_subnet(subnet_dict)

        # Clear bn statics
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        # BN Calibration
        self.model.train()
        for _ in range(max_train_iters):
            data, target = next(train_iter)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            del data, target, output

        # Start test
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        for _ in range(max_test_iters):
            data, target = next(val_iter)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            n = target.size(0)
            top1, top5 = accuracy(output, target, topk=(1, 5))
            top1_vacc.update(top1.item(), n)
            top5_vacc.update(top5.item(), n)

        return 100 - top1_vacc.avg
