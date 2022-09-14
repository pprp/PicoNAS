import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import get_model_complexity_info

import pplib.utils.utils as utils
from pplib.core.losses import CC, PairwiseRankLoss
from pplib.evaluator import MacroEvaluator
from pplib.nas.mutators import OneShotMutator
from pplib.predictor.pruners.measures.nwot import compute_nwot
from pplib.predictor.pruners.measures.zen import compute_zen_score
from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer
from .registry import register_trainer


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

        # init flops
        self._init_flops()

        if self.mutator is None:
            self.mutator = OneShotMutator()
            self.mutator.prepare_from_supernet(self.model)

        # evaluate the rank consistency
        self.evaluator = self._build_evaluator(
            num_sample=50, dataset=self.dataset)

        # pairwise rank loss
        self.pairwise_rankloss = PairwiseRankLoss()

        # distill loss
        self.distill_loss = CC()
        self.lambda_kd = 1000.0

        # type from kwargs
        if 'type' in kwargs:
            self.type = kwargs['type']
            assert self.type in {'random', 'hamming', 'adaptive'}
        else:
            self.type = None
        self.logger.info(f'Current type of macro trainer is: {self.type}.')

    def _build_evaluator(self, num_sample=50, dataset='cifar10'):
        return MacroEvaluator(
            self, num_sample=num_sample, type='test_acc', dataset=dataset)

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
            Distance between I and 1 2 is set to 2
            Distance between 1 and 2 is set to 0.5
            """
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                if v1 == v2:
                    continue
                if set([v1, v2]) == set(['1', '2']):
                    dist += 0.5
                elif set([v1, v2]) == set(['1', 'I']):
                    dist += 2
                elif set([v1, v2]) == set(['1', '2']):
                    dist += 2
            return dist

        assert type in {'random', 'hamming', 'adaptive'}
        if type == 'random':
            return self.mutator.random_subnet, self.mutator.random_subnet
        elif type == 'hamming':
            # mean: 9.312 std 1.77
            subnet1 = self.mutator.random_subnet
            max_iter = 10
            subnet2 = self.mutator.random_subnet

            # 调参，调大或者调小 (1) 9.3 (2) 7 (3) 10
            while hamming_dist(subnet1, subnet2) <= 10 and max_iter > 0:
                subnet2 = self.mutator.random_subnet
            if max_iter > 0:
                return subnet1, subnet2
            else:
                return subnet1, self.mutator.random_subnet
        elif type == 'adaptive':
            # mean: 7.789 std 2.90
            subnet1 = self.mutator.random_subnet
            max_iter = 10
            subnet2 = self.mutator.random_subnet

            # 调参，调大或者调小 (1) 7.789 (2) 5 (3) 8
            while adaptive_hamming_dist(subnet1, subnet2) < 8 and max_iter > 0:
                subnet2 = self.mutator.random_subnet
            if max_iter > 0:
                return subnet1, subnet2
            else:
                return subnet1, self.mutator.random_subnet

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

            # FairNAS
            # loss, outputs = self._forward_fairnas(batch_inputs)

            # Single Path One Shot
            # compute loss
            # loss, outputs = self.forward(batch_inputs, mode='loss')
            # backprop
            # loss.backward()

            # SPOS with pairwise rankloss
            loss, outputs = self._forward_pairwise_loss(batch_inputs)

            # spos with pairwise rankloss + cc distill
            # loss, outputs = self._forward_pairwise_loss_with_distill(
            # batch_inputs)

            # spos with multi-pair rank loss
            # loss, outputs = self._forward_multi_pairwise_loss(batch_inputs)

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
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
        return self.model(inputs)

    def _forward_fairnas(self, batch_inputs):
        """FairNAS Rules."""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        fair_list = self._generate_fair_list()

        for i in range(len(fair_list)):
            subnet_dict = fair_list[i]
            self.mutator.set_subnet(subnet_dict)
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, labels)
            loss.backward()
        return loss, outputs

    def _forward_pairwise_loss(self, batch_inputs):
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # sample the first subnet
        if self.type is None:
            subnet1, subnet2 = self.sample_subnet_by_type(type='adaptive')
        else:
            subnet1, subnet2 = self.sample_subnet_by_type(type=self.type)

        self.mutator.set_subnet(subnet1)
        outputs = self.model(inputs)
        loss1 = self._compute_loss(outputs, labels)
        loss1.backward()
        flops1 = self.get_subnet_flops(subnet1)

        # sample the second subnet
        self.mutator.set_subnet(subnet2)
        outputs = self.model(inputs)
        loss2 = self._compute_loss(outputs, labels)
        loss2.backward(retain_graph=True)
        flops2 = self.get_subnet_flops(subnet2)

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
        subnet1 = self.mutator.random_subnet
        self.mutator.set_subnet(subnet1)
        outputs, feat1 = self.model.forward_distill(inputs)
        loss1 = self._compute_loss(outputs, labels)
        flops1 = self.get_subnet_flops(subnet1)
        loss_list.append(loss1)

        # sample the second subnet
        subnet2 = self.mutator.random_subnet
        self.mutator.set_subnet(subnet2)
        outputs, feat2 = self.model.forward_distill(inputs)
        loss2 = self._compute_loss(outputs, labels)
        flops2 = self.get_subnet_flops(subnet2)
        loss_list.append(loss2)

        # pairwise rank loss
        # lambda settings:
        #       1. min(2, self.current_epoch/10.)
        #       2. 2 * np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs)

        loss3 = (2 *
                 np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs) *
                 self.pairwise_rankloss(flops1, flops2, loss1, loss2))
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
        subnet1 = self.mutator.random_subnet
        self.mutator.set_subnet(subnet1)
        outputs = self.model(inputs)
        loss1 = self._compute_loss(outputs, labels)
        flops1 = self.get_subnet_flops(subnet1)

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

    def _predict(self, batch_inputs, subnet_dict: Dict = None):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
        # forward pass
        if subnet_dict is None:
            rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(rand_subnet)
        else:
            self.mutator.set_subnet(subnet_dict)
        return self.model(inputs), labels

    def _generate_fair_list(self) -> List[Dict]:
        choices = ['I', '1', '2']
        length = 14

        all_list = [random.sample(choices, 3) for _ in range(length)]
        all_list = np.array(all_list).T

        return [dict(enumerate(all_list[i])) for i in range(len(choices))]

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
                    tag=f'{self.log_name}_macro',
                )

            self.train_loss_.append(tr_loss)
            self.val_loss_.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch: {epoch + 1}/{epochs} Time: {epoch_time} Train loss: {tr_loss} Val loss: {val_loss}'  # noqa: E501
            )

            if epoch % 5 == 0:
                kt, ps, sp = self.evaluator.compute_rank_consistency(
                    val_loader)
                self.writer.add_scalar(
                    'RANK/kendall_tau', kt, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/pearson', ps, global_step=self.current_epoch)
                self.writer.add_scalar(
                    'RANK/spearman', sp, global_step=self.current_epoch)

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

                # print every 20 iter
                if step % 20 == 0:
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
        #     f'Val loss: {loss.item()} Top1 acc: {top1_vacc.avg} Top5 acc: {top5_vacc.avg}'
        # )
        # val_loss / (step + 1), top1_vacc.avg, top5_vacc.avg
        return top1_vacc.avg

    def _init_flops(self):
        """generate flops."""
        self.model.eval()
        # Note 1: after this process, each module in self.model
        #       would have the __flops__ attribute.
        # Note 2: this function should be called before
        #       mutator.prepare_from_supernet()
        flops, params = get_model_complexity_info(self.model, self.input_shape)
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

    def get_subnet_zenscore(self, subnet_dict) -> float:
        """Calculate zenscore based on subnet dict."""
        import copy
        m = copy.deepcopy(self.model)
        o = OneShotMutator()
        o.prepare_from_supernet(m)
        o.set_subnet(subnet_dict)

        # for cifar10,cifar100,imagenet16
        score = compute_zen_score(
            net=m, inputs=torch.randn(4, 3, 32, 32), targets=None, repeat=5)
        del m
        del o
        return score

    def get_subnet_params(self, subnet_dict) -> float:
        """Calculate current subnet params based on mmcv."""
        subnet_params = 0
        for k, v in self.mutator.search_group.items():
            current_choice = subnet_dict[k]
            choice_params = 0
            for _, module in v[0]._candidate_ops[current_choice].named_modules(
            ):
                params = getattr(module, '__params__', 0)
                if params > 0:
                    choice_params += params
            subnet_params += choice_params
        return subnet_params

    def get_subnet_nwot(self, subnet_dict) -> float:
        """Calculate zenscore based on subnet dict."""
        import copy
        m = copy.deepcopy(self.model)
        o = OneShotMutator()
        o.prepare_from_supernet(m)
        o.set_subnet(subnet_dict)

        # for cifar10,cifar100,imagenet16
        score = compute_nwot(
            net=m,
            inputs=torch.randn(4, 3, 32, 32).to('cuda'),
            targets=torch.randn(4).to('cuda'))
        del m
        del o
        return score
