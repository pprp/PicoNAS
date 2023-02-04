import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import get_model_complexity_info

import piconas.utils.utils as utils
from piconas.core.losses import PairwiseRankLoss
from piconas.evaluator.nb301_evaluator import NB301Evaluator
from piconas.models.nasbench301 import OneShotNASBench301Network
from piconas.nas.mutables import OneShotOP
from piconas.nas.mutators import OneShotMutator
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer
from .registry import register_trainer


@register_trainer
class NB301Trainer(BaseTrainer):
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
        model: OneShotNASBench301Network,
        mutator: OneShotMutator,
        optimizer=None,
        criterion=None,
        scheduler=None,
        device: torch.device = torch.device('cuda'),
        log_name='nasbench301',
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
            **kwargs)

        # init flops
        self._init_flops()

        if self.mutator is None:
            # Note: use alias to build search group
            self.mutator = OneShotMutator(with_alias=True)
            self.mutator.prepare_from_supernet(model)

        # evaluate the rank consistency
        self.evaluator = self._build_evaluator(
            num_sample=50, dataset=self.dataset)

        # pairwise rank loss
        self.pairwise_rankloss = PairwiseRankLoss()

        # record current rand_subnet
        self.rand_subnet = None

        # Forward Specific Subnet flag
        #  => is_specific is True: cooperate with SH
        #  => is_specific is False: normal mode
        self.is_specific = False

        # type from kwargs can be random, hamming, adaptive
        if 'type' in kwargs:
            self.type = kwargs['type']
            assert self.type in {
                'random', 'hamming', 'adaptive', 'uniform', 'fair'
            }
        else:
            self.type = None
        self.logger.info(f'Current type of nb301 trainer is: {self.type}.')

    def _build_evaluator(self, num_sample=50, dataset='cifar10'):
        return NB301Evaluator(self, num_sample, dataset)

    def sample_subnet_by_type(self, type: str = 'random') -> List[Dict]:
        """Return two subnets based on ``type``.

        Type:
            - ``random``: random sample two subnet.
            - ``hamming``: sample subnet with hamming distance.
            - ``adaptive``: sample subnet with adaptive hamming distance.
        """

        def hamming_dist(dct1, dct2):
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                dist += 1 if v1 != v2 else 0
            return dist

        def adaptive_hamming_dist(dct1, dct2):
            """
            Distance between conv is set to 0.5
            Distance between conv and other is set to 2
            Distance between other and other is set to 0.5
            """
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                if v1 == v2:
                    continue
                if 'conv' in v1 and 'conv' in v2:
                    dist += 0.5
                elif 'conv' in v1 and ('skip' in v2 or 'pool' in v2):
                    dist += 2
                elif 'conv' in v2 and ('skip' in v1 or 'pool' in v1):
                    dist += 2
                elif 'skip' in v1 and 'pool' in v2:
                    dist += 0.5
                elif 'skip' in v2 and 'pool' in v1:
                    dist += 0.5
                else:
                    raise NotImplementedError(f'v1: {v1} v2: {v2}')
            return dist

        assert type in {'random', 'hamming', 'adaptive'}
        if type == 'random':
            return self.mutator.random_subnet, self.mutator.random_subnet
        elif type == 'hamming':
            # mean: 4.5 std 1.06
            subnet1 = self.mutator.random_subnet
            max_iter = 10
            subnet2 = self.mutator.random_subnet
            while hamming_dist(subnet1, subnet2) < 4.5 and max_iter > 0:
                subnet2 = self.mutator.random_subnet
            if max_iter > 0:
                return subnet1, subnet2
            else:
                return subnet1, self.mutator.random_subnet
        elif type == 'adaptive':
            # mean: 6.7 std 2.23
            subnet1 = self.mutator.random_subnet
            max_iter = 10
            subnet2 = self.mutator.random_subnet

            # 调参，调大或者调小 (1) 6.7 (2) 5 (3) 11
            while adaptive_hamming_dist(subnet1,
                                        subnet2) < 6.7 and max_iter > 0:
                subnet2 = self.mutator.random_subnet
            if max_iter > 0:
                return subnet1, subnet2
            else:
                return subnet1, self.mutator.random_subnet

    def _generate_fair_lists(self) -> List[Dict]:
        search_group = self.mutator.search_group
        fair_lists = []

        choices_dict = dict()
        num_choices = -1
        for group_id, modules in search_group.items():
            choices = modules[0].choices
            choices_dict[group_id] = random.sample(choices, len(choices))
            num_choices = len(choices)

        for i in range(num_choices):
            current_dict = dict()
            for k, v in choices_dict.items():
                current_dict[k] = v[i]
            fair_lists.append(current_dict)
        return fair_lists

    def _train(self, loader):
        self.model.train()

        train_loss = 0.0
        top1_tacc = AvgrageMeter()
        top5_tacc = AvgrageMeter()

        for step, batch_inputs in enumerate(loader):
            # get image and labels
            inputs, labels = batch_inputs
            inputs = self._to_device(inputs, self.device)
            labels = self._to_device(labels, self.device)

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            if self.type in {'uniform', 'fair'}:
                if self.type == 'uniform':
                    loss, outputs = self._forward_uniform(batch_inputs)
                elif self.type == 'fair':
                    loss, outputs = self._forward_fairnas(batch_inputs)
            else:
                loss, outputs = self._forward_pairwise_loss(batch_inputs)

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
                    f'Step: {step:03} Train loss: {loss.item():.4f} Top1 acc: {top1_tacc.avg:.3f} Top5 acc: {top5_tacc.avg:.3f}'
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

        if self.is_specific:
            return self.model(inputs)

        # forward pass
        if self.searching:
            self.rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(self.rand_subnet)
        return self.model(inputs)

    def _predict(self, batch_inputs, subnet_dict: Dict = None):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        if self.is_specific:
            return self.model(inputs), labels

        # forward pass
        if subnet_dict is None:
            self.rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(self.rand_subnet)
        else:
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
            tr_loss, top1_tacc, top5_tacc = self._train(train_loader)

            # validate
            val_loss, top1_vacc, top5_vacc = self._validate(val_loader)

            # save ckpt
            if epoch % 10 == 0:
                utils.save_checkpoint(
                    {'state_dict': self.model.state_dict()},
                    self.log_name,
                    epoch + 1,
                    tag=f'{self.log_name}_nb301',
                )

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch: {epoch + 1}/{epochs} Time: {epoch_time} Train loss: {tr_loss} Val loss: {val_loss}'  # noqa: E501
            )

            if epoch % 5 == 0:
                assert self.evaluator is not None
                # BWR@K, P@tbk
                kt, ps, sp, rd, minn_at_ks, patks, cpr = self.evaluator.compute_rank_consistency(
                    val_loader, self.mutator)
                self.writer.add_scalar(
                    'RANK/kendall_tau', kt, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/pearson', ps, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/spearman', sp, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/cpr', cpr, global_step=self.current_epoch)

                if isinstance(rd, list):
                    for i, r in enumerate(rd):
                        self.writer.add_scalar(
                            f'ANALYSE/rank_diff_{(i+1)*20}%',
                            r,
                            global_step=self.current_epoch)
                else:
                    self.writer.add_scalar(
                        'ANALYSE/rank_diff',
                        rd,
                        global_step=self.current_epoch)

                for k, minn, brk, maxn, wrk in minn_at_ks:
                    # self.writer.add_scalar(
                    #     f'ANALYSE/oneshot_{k}_minn',
                    #     minn,
                    #     global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{k}_BR@K',
                        brk,
                        global_step=self.current_epoch)
                    # self.writer.add_scalar(
                    #     f'ANALYSE/oneshot_{k}_maxn',
                    #     maxn,
                    #     global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{k}_WR@K',
                        wrk,
                        global_step=self.current_epoch)

                for ratio, k, p_at_topk, p_at_bk, kd_at_topk, kd_at_bk in patks:
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_P@topK',
                        p_at_topk,
                        global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_P@bottomK',
                        p_at_bk,
                        global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_KD@topK',
                        kd_at_topk,
                        global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_KD@bottomK',
                        kd_at_bk,
                        global_step=self.current_epoch)

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
            if isinstance(v[0], OneShotOP):
                for _, module in v[0]._candidate_ops[
                        current_choice].named_modules():
                    flops = getattr(module, '__flops__', 0)
                    if flops > 0:
                        choice_flops += flops
            # print(f'k: {k} choice: {current_choice} flops: {choice_flops}')
            subnet_flops += choice_flops
        return subnet_flops

    def get_subnet_predictive(self,
                              subnet_dict,
                              dataloader,
                              measure_name='nwot') -> float:
        """Calculate zenscore based on subnet dict."""
        import copy
        m = copy.deepcopy(self.model)
        o = OneShotMutator(with_alias=True)
        o.prepare_from_supernet(m)
        o.set_subnet(subnet_dict)
        dataload_info = ['random', 3, self.num_classes]

        # for cifar10,cifar100,imagenet16
        score = find_measures(
            self.trainer.model,
            dataloader,
            dataload_info=dataload_info,
            measure_names=measure_name,
            loss_fn=F.cross_entropy,
            device=self.trainer.device)
        del m
        del o
        return score

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
            try:
                data, target = next(train_iter)
            except:
                del train_iter
                train_iter = iter(train_loader)
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

    def _forward_uniform(self, batch_inputs):
        """Single Path One Shot."""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        self.mutator.set_subnet(self.mutator.random_subnet)
        outputs = self.model(inputs)
        loss = self._compute_loss(outputs, labels)
        loss.backward()
        return loss, outputs

    def _forward_fairnas(self, batch_inputs):
        """FairNAS Rules."""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        fair_dicts = self._generate_fair_lists()

        loss_list = []

        for i in range(len(fair_dicts)):
            self.rand_subnet = fair_dicts[i]
            self.mutator.set_subnet(self.rand_subnet)
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, labels)
            loss_list.append(loss)

        sum_loss = sum(loss_list)
        sum_loss.backward()
        return sum_loss, outputs

    def fit_specific(self, train_loader, val_loader, epochs,
                     subnet_cfg: dict) -> None:
        """Fits. High Level API
        Fit the model using the given loaders for the given number
        of epochs.
        """
        # change the flag
        self.is_specific = True
        self.mutator.set_subnet(subnet_cfg)

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
            tr_loss, top1_tacc, top5_tacc = self._train(train_loader)

            # validate
            val_loss, top1_vacc, top5_vacc = self._validate_specific(
                val_loader)

            # save ckpt
            if epoch % 10 == 0:
                utils.save_checkpoint(
                    {'state_dict': self.model.state_dict()},
                    self.log_name,
                    epoch + 1,
                    tag=f'{self.log_name}_nb301',
                )

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch: {epoch + 1}/{epochs} Time: {epoch_time} Train loss: {tr_loss} Val loss: {val_loss}'  # noqa: E501
            )

            if epoch % 5 == 0:
                assert self.evaluator is not None
                # BWR@K, P@tbk
                kt, ps, sp, rd, minn_at_ks, patks, cpr = self.evaluator.compute_rank_consistency(
                    val_loader, self.mutator)
                self.writer.add_scalar(
                    'RANK/kendall_tau', kt, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/pearson', ps, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/spearman', sp, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/cpr', cpr, global_step=self.current_epoch)

                if isinstance(rd, list):
                    for i, r in enumerate(rd):
                        self.writer.add_scalar(
                            f'ANALYSE/rank_diff_{(i+1)*20}%',
                            r,
                            global_step=self.current_epoch)
                else:
                    self.writer.add_scalar(
                        'ANALYSE/rank_diff',
                        rd,
                        global_step=self.current_epoch)

                for k, minn, brk, maxn, wrk in minn_at_ks:
                    # self.writer.add_scalar(
                    #     f'ANALYSE/oneshot_{k}_minn',
                    #     minn,
                    #     global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{k}_BR@K',
                        brk,
                        global_step=self.current_epoch)
                    # self.writer.add_scalar(
                    #     f'ANALYSE/oneshot_{k}_maxn',
                    #     maxn,
                    #     global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{k}_WR@K',
                        wrk,
                        global_step=self.current_epoch)

                for ratio, k, p_at_topk, p_at_bk, kd_at_topk, kd_at_bk in patks:
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_P@topK',
                        p_at_topk,
                        global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_P@bottomK',
                        p_at_bk,
                        global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_KD@topK',
                        kd_at_topk,
                        global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_KD@bottomK',
                        kd_at_bk,
                        global_step=self.current_epoch)

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

    def _forward_pairwise_loss(self, batch_inputs):
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # random, hamming, adaptive
        if self.type is None:
            subnet1, subnet2 = self.sample_subnet_by_type(type='random')
        else:
            subnet1, subnet2 = self.sample_subnet_by_type(type=self.type)

        # sample the first subnet
        self.mutator.set_subnet(subnet1)
        outputs = self.model(inputs)
        loss1 = self._compute_loss(outputs, labels)
        loss1.backward()
        flops1 = self.get_subnet_flops(subnet1)
        # nwot1 = self.get_subnet_nwot(subnet1)

        # sample the second subnet
        self.mutator.set_subnet(subnet2)
        outputs = self.model(inputs)
        loss2 = self._compute_loss(outputs, labels)
        loss2.backward(retain_graph=True)
        flops2 = self.get_subnet_flops(subnet2)
        # nwot2 = self.get_subnet_nwot(subnet2)

        # pairwise rank loss
        # lambda settings:
        #       1. min(2, self.current_epoch/10.)
        #       2. 2 * np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs)

        loss3 = (2 *
                 np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs) *
                 self.pairwise_rankloss(flops1, flops2, loss1, loss2))
        loss3.backward()

        return loss2, outputs

    def _forward_pairwise_loss_with_distill(self, batch_inputs):
        """
        Policy:
            1. use larger flops model as teacher
            2. use lower loss model as teacher
        """
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        loss_list = []

        # sample the first subnet
        rand_subnet1 = self.mutator.random_subnet
        self.mutator.set_subnet(rand_subnet1)
        outputs, feat1 = self.model.forward_distill(inputs)
        loss1 = self._compute_loss(outputs, labels)
        flops1 = self.get_subnet_flops(rand_subnet1)
        loss_list.append(loss1)

        # sample the second subnet
        rand_subnet2 = self.mutator.random_subnet
        self.mutator.set_subnet(rand_subnet2)
        outputs, feat2 = self.model.forward_distill(inputs)
        loss2 = self._compute_loss(outputs, labels)
        flops2 = self.get_subnet_flops(rand_subnet2)
        loss_list.append(loss2)

        # pairwise rank loss
        # lambda settings:
        #       1. min(2, self.current_epoch/10.)
        #       2. 2 * np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs)

        loss3 = self._lambda * self.pairwise_rankloss(flops1, flops2, loss1,
                                                      loss2)
        loss_list.append(loss3)

        # distill loss
        if loss2 > loss1:
            loss4 = self.distill_loss(
                feat_s=feat2, feat_t=feat1) * self.lambda_kd
        else:
            loss4 = self.distill_loss(
                feat_s=feat1, feat_t=feat2) * self.lambda_kd
        loss_list.append(loss4)

        loss = sum(loss_list)
        loss.backward()

        return loss, outputs

    def _forward_multi_pairwise_loss(self, batch_inputs):
        num_pairs = 4

        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # sample the first subnet
        rand_subnet1 = self.mutator.random_subnet
        self.mutator.set_subnet(rand_subnet1)
        outputs = self.model(inputs)
        loss1 = self._compute_loss(outputs, labels)
        flops1 = self.get_subnet_flops(rand_subnet1)

        subnet_list = []
        loss_list = []
        flops_list = []

        for _ in range(num_pairs):
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, labels)
            flops = self.get_subnet_flops(rand_subnet)

            subnet_list.append(rand_subnet)
            loss_list.append(loss)
            flops_list.append(flops)

        rank_loss_list = []

        for i in range(1, num_pairs):
            for j in range(i):
                flops1, flops2 = flops_list[i], flops_list[j]
                loss1, loss2 = loss_list[i], loss_list[j]
                tmp_rank_loss = self.pairwise_rankloss(flops1, flops2, loss1,
                                                       loss2)

                rank_loss_list.append(tmp_rank_loss)

        sum_loss = sum(loss_list) + sum(rank_loss_list)
        sum_loss.backward()

        return sum_loss, outputs
