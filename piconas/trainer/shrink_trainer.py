import copy
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import lr_scheduler

import piconas.utils.utils as utils
from piconas.core.losses import KLDivergence, PairwiseRankLoss
from piconas.evaluator.nb201_evaluator import NB201Evaluator
from piconas.models.nasbench201 import OneShotNASBench201Network
from piconas.nas.mutators import OneShotMutator
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.flops_counter import get_model_complexity_info
from piconas.utils.utils import AvgrageMeter, accuracy
from piconas.utils.weight_init import constant_init, normal_init
from .base import BaseTrainer
from .registry import register_trainer


class NB201Shrinker(object):
    """Search Space Shrinker for NB201.

    Args:
        trainer (_type_): _description_
    """

    def __init__(
        self, trainer: BaseTrainer, sample_num: int = 200, per_stage_drop_num: int = 1
    ):
        self.trainer = trainer
        self.mutator = trainer.mutator
        self.sample_num = sample_num
        self.per_stage_drop_num = per_stage_drop_num

    def expand_operator(self, extend_operators, vis_dict_slice):
        """each operator is ranked according to its metric score."""

        # sort by
        extend_operators.sort(
            key=lambda x: vis_dict_slice[x]['metric'], reverse=False)

        # drop operators whose ranking fall at the tail.
        num = 0
        for i, operator in enumerate(extend_operators):
            # score of current operator is lowest and should be pruned.
            id, choice = operator
            drop_legal = False

            # expand choice
            expand_ops = []

            # at lease one operator should be reserved for each layer.
            for j in range(i + 1, len(extend_operators)):
                # get current extended operations
                idx_, choice_ = extend_operators[j]
                if idx_ == id:
                    # if find a better operator, we can remove lowest one.
                    drop_legal = True
                    expand_ops.append(extend_operators[j])

            if drop_legal:
                expand_op = random.choice(expand_ops)
                self.trainer.logger.info(
                    f'no.{num + 1} expand_op={expand_op} metric={vis_dict_slice[expand_op]["metric"]}'
                )

                groups = self.mutator.search_group[id]
                for item in groups:
                    # expand search space
                    item.expand_choice(expand_op[1])
                num += 1
            if num >= self.per_stage_drop_num:
                break
        return expand_ops

    def drop_operator(self, extend_operators, vis_dict_slice):
        """each operator is ranked according to its metric score."""

        # sort by
        extend_operators.sort(
            key=lambda x: vis_dict_slice[x]['metric'], reverse=False)

        # drop operators whose ranking fall at the tail.
        num, drop_ops = 0, []
        for i, operator in enumerate(extend_operators):
            # score of current operator is lowest and should be pruned.
            id, choice = operator
            drop_legal = False

            # at lease one operator should be reserved for each layer.
            for j in range(i + 1, len(extend_operators)):
                # get current extended operations
                idx_, choice_ = extend_operators[j]
                if idx_ == id:
                    # if find a better operator, we can remove lowest one.
                    drop_legal = True

            if drop_legal:
                self.trainer.logger.info(
                    f'no.{num + 1} drop_op={operator} metric={vis_dict_slice[operator]["metric"]}'
                )

                drop_ops.append(operator)
                # remove from search space
                groups = self.mutator.search_group[id]
                for item in groups:
                    item.shrink_choice(choice)
                num += 1
            if num >= self.per_stage_drop_num:
                break
        return drop_ops

    def compute_score(
        self, extend_operators, vis_dict_slice, vis_dict, train_loader, val_loader
    ):
        """
        1. Random sample `num` of architectures extended by some operators.
        2. Compute metrics of all candidate architectures.
        3. Caculate sum of metrics for each operator.
        """
        candidates = []
        # step 1: randomly sample extended operators.
        assert len(extend_operators) > 0
        for operator in extend_operators:
            info = vis_dict_slice[operator]
            if self.sample_num - len(info['cand_pool']) > 0:
                num = self.sample_num - len(info['cand_pool'])
                candidate_subnets = self.get_random_extend(
                    num, operator, vis_dict)

                for subnet in candidate_subnets:
                    for id, choice in subnet.items():
                        extend_operator_ = (id, choice)
                        if extend_operator_ in vis_dict_slice:
                            info = vis_dict_slice[extend_operator_]
                            if subnet not in info['cand_pool']:
                                info['cand_pool'].append(subnet)
                    if subnet not in candidates:
                        candidates.append(subnet)

        # step 2: compute metrics of all candidate subnets
        for subnet in candidates:
            info = vis_dict[str(subnet)]
            # info['metric'] = self.trainer.get_subnet_nwot(subnet)
            # info['metric'] = self.trainer.get_subnet_flops(subnet)
            info['metric'] = self.trainer.get_subnet_acc(
                subnet, train_loader, val_loader
            )

        # step 3: calculate sum of angles for each operator
        for subnet in candidates:
            info = vis_dict[str(subnet)]
            for id, choice in subnet.items():
                extend_operator_ = (id, choice)
                if extend_operator_ in vis_dict_slice:
                    slice_info = vis_dict_slice[extend_operator_]
                    if (
                        subnet in slice_info['cand_pool']
                        and slice_info['count'] < self.sample_num
                    ):
                        slice_info['metric'] += info['metric']
                        slice_info['count'] += 1

        # step 4: compute scores of all candidate operators
        for operator in extend_operators:
            if vis_dict_slice[operator]['count'] > 0:
                vis_dict_slice[operator]['metric'] /= vis_dict_slice[operator]['count']

    def get_random_extend(self, num: int, operator: Tuple[int, str], vis_dict):
        def get_extend_subnet(operator):
            """Here we do not consider for skip connection as anglenas."""
            id, choice = operator
            extend_subnet = dict()
            for id_i, ops in self.mutator.search_group.items():
                if id_i == id and choice is not None:
                    extend_subnet[id] = choice
                else:
                    if len(ops[0].choices) == 1:
                        extend_subnet[id_i] = ops[0].choices[0]
                    else:
                        # random sample
                        sampled_choice = random.choice(ops[0].choices)
                        extend_subnet[id_i] = sampled_choice
            return extend_subnet

        max_iter = num * 100
        candidate_subnets = []
        i = 0
        while i < num and max_iter > 0:
            max_iter -= 1
            subnet = get_extend_subnet(operator)
            if not self.is_legal(subnet, vis_dict):
                continue
            candidate_subnets.append(subnet)
            i += 1
        return candidate_subnets

    def is_legal(self, subnet=None, vis_dict=None) -> bool:
        """make sure each subnet is sampled only once."""
        if subnet is None:
            return False
        if str(subnet) not in vis_dict:
            vis_dict[str(subnet)] = dict()
        info = vis_dict[str(subnet)]
        if 'visited' in info:
            return False
        info['visited'] = True
        vis_dict[str(subnet)] = info
        return True

    def shrink(self, train_loader, val_loader):
        # split every single operation in each layer.
        # travese all of search group.
        vis_dict_slice: Dict[tuple, dict] = dict()
        vis_dict: Dict[str, dict] = dict()

        # every single operation
        extend_operators = []
        for id, ops in self.mutator.search_group.items():
            # each group has same type of candidate operations.
            for choice in ops[0].choices:
                operator = id, choice
                vis_dict_slice[operator] = dict()
                vis_dict_slice[operator]['count'] = 0
                vis_dict_slice[operator]['metric'] = 0
                # for example angle
                vis_dict_slice[operator]['cand_pool'] = []
                extend_operators.append(operator)

        self.compute_score(
            extend_operators, vis_dict_slice, vis_dict, train_loader, val_loader
        )
        drop_ops = self.drop_operator(extend_operators, vis_dict_slice)
        self.trainer.logger.info(f'drop ops: {drop_ops}')

    def expand(self, train_loader, val_loader):
        # split every single operation in each layer.
        # travese all of search group.
        vis_dict_slice: Dict[tuple, dict] = dict()
        vis_dict: Dict[str, dict] = dict()

        # every single operation
        extend_operators = []
        for id, ops in self.mutator.search_group.items():
            # each group has same type of candidate operations.
            for choice in ops[0].choices:
                operator = id, choice
                vis_dict_slice[operator] = dict()
                vis_dict_slice[operator]['count'] = 0
                vis_dict_slice[operator]['metric'] = 0
                # for example angle
                vis_dict_slice[operator]['cand_pool'] = []
                extend_operators.append(operator)

        self.compute_score(
            extend_operators, vis_dict_slice, vis_dict, train_loader, val_loader
        )
        expand_ops = self.expand_operator(extend_operators, vis_dict_slice)
        self.trainer.logger.info(f'expand ops: {expand_ops}')


@register_trainer
class NB201ShrinkTrainer(BaseTrainer):
    """Trainer for modifying search space.

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
        model: OneShotNASBench201Network,
        mutator: OneShotMutator,
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

        # init flops
        self._init_flops()

        if self.mutator is None:
            # Note: use alias to build search group
            self.mutator = OneShotMutator(with_alias=True)
            self.mutator.prepare_from_supernet(model)

        # evaluate the rank consistency
        self.evaluator = self._build_evaluator(
            num_sample=1000, dataset=self.dataset)

        # pairwise rank loss
        self.pairwise_rankloss = PairwiseRankLoss()

        # record current rand_subnet
        self.rand_subnet = None

        # Forward Specific Subnet flag
        #  => is_specific is True: cooperate with SH
        #  => is_specific is False: normal mode
        self.is_specific = False

        self.max_subnet = {
            0: 'nor_conv_3x3',
            1: 'nor_conv_3x3',
            2: 'nor_conv_3x3',
            3: 'nor_conv_3x3',
            4: 'nor_conv_3x3',
            5: 'nor_conv_3x3',
        }

        self.min_subnet = {
            0: 'nor_conv_3x3',
            1: 'nor_conv_3x3',
            2: 'skip_connect',
            3: 'skip_connect',
            4: 'skip_connect',
            5: 'skip_connect',
        }

        # type from kwargs can be random, hamming, adaptive
        if 'type' in kwargs:
            self.type = kwargs['type']
            assert self.type in {
                'random',
                'hamming',
                'adaptive',
                'uniform',
                'fair',
                'sandwich',
                'zenscore',
                'flops',
                'params',
                'nwot',
            }
        else:
            self.type = None
        self.logger.info(f'Current type of nb201 trainer is: {self.type}.')

        self.shrinker = NB201Shrinker(self)
        self.expand_times = kwargs['expand_times'] if 'expand_times' in kwargs else 6
        self.shrink_times = kwargs['shrink_times'] if 'shrink_times' in kwargs else 3
        self.every_n_epochs = (
            kwargs['every_n_epochs'] if 'every_n_epochs' in kwargs else 5
        )

        self.kl_loss = KLDivergence(loss_weight=1)

    def _build_evaluator(self, num_sample=50, dataset='cifar10'):
        return NB201Evaluator(self, num_sample, dataset)

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
            while adaptive_hamming_dist(subnet1, subnet2) < 6.7 and max_iter > 0:
                subnet2 = self.mutator.random_subnet
            if max_iter > 0:
                return subnet1, subnet2
            else:
                return subnet1, self.mutator.random_subnet

    def sample_subnet_by_policy(
        self, policy: str = 'balanced', n_samples: int = 3
    ) -> Dict:
        assert policy in {'zenscore', 'flops', 'params', 'nwot'}
        n_subnets = [self.mutator.random_subnet for _ in range(n_samples)]

        def minmaxscaler(n_list: Tensor) -> Tensor:
            min_n = torch.min(n_list)
            max_n = torch.max(n_list)
            return (n_list - min_n) / max_n - min_n

        if policy == 'flops':
            n_flops = torch.tensor([self.get_subnet_flops(i)
                                   for i in n_subnets])
            res = minmaxscaler(n_flops)
            res = F.softmax(res, dim=0)
            # Find the max
            max_idx = res.argmax()
            # Get corresponding subnet
            subnet = n_subnets[max_idx]

        elif policy == 'params':
            n_params = torch.tensor(
                [self.get_subnet_params(i) for i in n_subnets])
            res = minmaxscaler(n_params)
            res = F.softmax(res, dim=0)
            # Find the max
            max_idx = res.argmax()
            subnet = n_subnets[max_idx]

        elif policy == 'zenscore':
            n_zenscore = torch.tensor(
                [self.get_subnet_zenscore(i) for i in n_subnets])
            res = minmaxscaler(n_zenscore)
            res = F.softmax(res, dim=0)
            # Find the max
            max_idx = res.argmax()
            subnet = n_subnets[max_idx]
        elif policy == 'nwot':
            n_nwot = torch.tensor([self.get_subnet_nwot(i) for i in n_subnets])
            res = minmaxscaler(n_nwot)
            res = F.softmax(res, dim=0)
            # Find the max
            max_idx = res.argmax()
            subnet = n_subnets[max_idx]

        return subnet

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

    def _train(self, train_loader, val_loader):
        self.model.train()

        def calc_search_space_size(search_group):
            ss_size = 1
            for _, v in search_group.items():
                ss_size *= len(v[0]._candidate_ops)
            return ss_size

        train_loss = 0.0
        top1_tacc = AvgrageMeter()
        top5_tacc = AvgrageMeter()

        for step, batch_inputs in enumerate(train_loader):
            # get image and labels
            inputs, labels = batch_inputs
            inputs = self._to_device(inputs, self.device)
            labels = self._to_device(labels, self.device)

            # remove gradient from previous passes
            self.optimizer.zero_grad()

            if self.type in {'uniform', 'fair', 'sandwich'}:
                if self.type == 'uniform':
                    loss, outputs = self._forward_uniform(batch_inputs)
                elif self.type == 'fair':
                    loss, outputs = self._forward_fairnas(batch_inputs)
                elif self.type == 'sandwich':
                    loss, outputs = self._forward_sandwich(batch_inputs)
            elif self.type in {'zenscore', 'flops', 'params', 'nwot'}:
                loss, outputs = self._forward_balanced(
                    batch_inputs, policy=self.type)
            else:
                loss, outputs = self._forward_pairwise_loss(batch_inputs)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

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
                f' Top5 acc: {top5_vacc.avg}'
            )
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

            if (
                self.expand_times > 0
                and (self.current_epoch + 1) % self.every_n_epochs == 0
                and self.current_epoch > 100
            ):
                self.shrinker.expand(train_loader, val_loader)
                self.expand_times -= 1

            if (
                self.shrink_times > 0
                and (self.current_epoch + 1) % self.every_n_epochs == 0
                and self.current_epoch > 100
            ):
                self.shrinker.shrink(train_loader, val_loader)
                self.shrink_times -= 1

            if (self.current_epoch < 100 and epoch % 10 == 0) or (
                self.current_epoch >= 100 and epoch % 100 == 0
            ):
                assert self.evaluator is not None
                # BWR@K, P@tbk
                (
                    kt,
                    ps,
                    sp,
                    rd,
                    minn_at_ks,
                    patks,
                    cpr,
                ) = self.evaluator.compute_rank_consistency(val_loader, self.mutator)
                self.writer.add_scalar(
                    'RANK/kendall_tau', kt, global_step=self.current_epoch
                )
                self.writer.add_scalar(
                    'RANK/pearson', ps, global_step=self.current_epoch
                )
                self.writer.add_scalar(
                    'RANK/spearman', sp, global_step=self.current_epoch
                )
                self.writer.add_scalar(
                    'RANK/cpr', cpr, global_step=self.current_epoch)

                if isinstance(rd, list):
                    for i, r in enumerate(rd):
                        self.writer.add_scalar(
                            f'ANALYSE/rank_diff_{(i+1)*20}%',
                            r,
                            global_step=self.current_epoch,
                        )
                else:
                    self.writer.add_scalar(
                        'ANALYSE/rank_diff', rd, global_step=self.current_epoch
                    )

                for k, minn, brk, maxn, wrk in minn_at_ks:
                    # self.writer.add_scalar(
                    #     f'ANALYSE/oneshot_{k}_minn',
                    #     minn,
                    #     global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{k}_BR@K', brk, global_step=self.current_epoch
                    )
                    # self.writer.add_scalar(
                    #     f'ANALYSE/oneshot_{k}_maxn',
                    #     maxn,
                    #     global_step=self.current_epoch)
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{k}_WR@K', wrk, global_step=self.current_epoch
                    )

                for ratio, k, p_at_topk, p_at_bk, kd_at_topk, kd_at_bk in patks:
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_P@topK',
                        p_at_topk,
                        global_step=self.current_epoch,
                    )
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_P@bottomK',
                        p_at_bk,
                        global_step=self.current_epoch,
                    )
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_KD@topK',
                        kd_at_topk,
                        global_step=self.current_epoch,
                    )
                    self.writer.add_scalar(
                        f'ANALYSE/oneshot_{ratio}_KD@bottomK',
                        kd_at_bk,
                        global_step=self.current_epoch,
                    )

            self.writer.add_scalar(
                'EPOCH_LOSS/train_epoch_loss', tr_loss, global_step=self.current_epoch
            )
            self.writer.add_scalar(
                'EPOCH_LOSS/valid_epoch_loss', val_loss, global_step=self.current_epoch
            )

            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

        total_time = time.time() - total_start_time

        # final message
        self.logger.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )

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
            current_choice = subnet_dict[k]
            choice_flops = 0
            for _, module in v[0]._candidate_ops[current_choice].named_modules():
                flops = getattr(module, '__flops__', 0)
                if flops > 0:
                    choice_flops += flops
            # print(f'k: {k} choice: {current_choice} flops: {choice_flops}')
            subnet_flops += choice_flops
        return subnet_flops

    def init_weights(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, val=1, bias=0.0001)
                nn.init.constant_(m.running_mean, 0)

    def get_subnet_predictive(
        self, subnet_dict, dataloader, measure_name='nwot'
    ) -> float:
        """Calculate zenscore based on subnet dict."""
        o = OneShotMutator(with_alias=True)
        copy_model = copy.deepcopy(self.model)
        self.init_weights(copy_model)
        o.prepare_from_supernet(copy_model)
        o.set_subnet(subnet_dict)

        dataload_info = ['random', 3, self.num_classes]

        score = find_measures(
            copy_model,
            dataloader,
            dataload_info=dataload_info,
            measure_names=measure_name,
            loss_fn=F.cross_entropy,
            device=self.trainer.device,
        )

        del o
        del copy_model
        return score

    def get_subnet_error(
        self, subnet_dict: Dict, train_loader=None, val_loader=None
    ) -> float:
        """Calculate the subnet of validation error.
        Including:
        1. BN calibration
        2. Start test
        """
        # Process dataloader
        assert train_loader is not None
        assert val_loader is not None

        # Info about dataloader
        iter_train = iter(train_loader)
        max_train_iters = 200

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
                data, target = next(iter_train)
            except:
                del iter_train
                iter_train = iter(train_loader)
                data, target = next(iter_train)

            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            del data, target, output

        # Start test
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        for data, target in range(val_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            n = target.size(0)
            top1, top5 = accuracy(output, target, topk=(1, 5))
            top1_vacc.update(top1.item(), n)
            top5_vacc.update(top5.item(), n)

        return 100 - top1_vacc.avg

    def get_subnet_acc(
        self, subnet_dict: Dict, train_loader=None, val_loader=None
    ) -> float:
        """Calculate the subnet of validation error.
        Including:
        1. BN calibration
        2. Start test
        """
        # Process dataloader
        assert train_loader is not None
        assert val_loader is not None

        # Info about dataloader
        max_train_iters = 200
        iter_train = iter(train_loader)

        self.mutator.set_subnet(subnet_dict)

        # Clear bn statics
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        # BN Calibration
        self.model.train()
        while max_train_iters > 0:
            max_train_iters -= 1
            try:
                data, target = next(iter_train)
            except:
                del iter_train
                iter_train = iter(train_loader)
                data, target = next(iter_train)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            del data, target, output

        # Start test
        top1_vacc = AvgrageMeter()
        top5_vacc = AvgrageMeter()

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            n = target.size(0)
            top1, top5 = accuracy(output, target, topk=(1, 5))
            top1_vacc.update(top1.item(), n)
            top5_vacc.update(top5.item(), n)

        return top1_vacc.avg

    def _forward_balanced(self, batch_inputs, policy='zenscore'):
        """Balanced Sampling Rules.
        Policy can be `zenscore`, `flops`, `params`, `nwot`
        """
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        subnet = self.sample_subnet_by_policy(policy=policy)
        self.mutator.set_subnet(subnet)
        outputs = self.model(inputs)
        loss = self._compute_loss(outputs, labels)
        loss.backward()
        return loss, outputs

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

        loss3 = (
            2
            * np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs)
            * self.pairwise_rankloss(flops1, flops2, loss1, loss2)
        )
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

        loss3 = self._lambda * self.pairwise_rankloss(
            flops1, flops2, loss1, loss2)
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
                tmp_rank_loss = self.pairwise_rankloss(
                    flops1, flops2, loss1, loss2)

                rank_loss_list.append(tmp_rank_loss)

        sum_loss = sum(loss_list) + sum(rank_loss_list)
        sum_loss.backward()

        return sum_loss, outputs

    def _forward_sandwich(self, batch_inputs):
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)

        # execuate full-network
        self.mutator.set_subnet(self.max_subnet)
        teacher_output = self.model(inputs)
        loss = self._compute_loss(teacher_output, labels)
        loss.backward()

        # execuate min-network
        self.mutator.set_subnet(self.min_subnet)
        student_output = self.model(inputs)
        loss = self.kl_loss(student_output, teacher_output.detach())
        loss.backward()

        # random sample two subnet
        for _ in range(2):
            self.mutator.set_subnet(self.mutator.random_subnet)
            student_output = self.model(inputs)
            loss = self.kl_loss(student_output, teacher_output.detach())
            loss.backward()

        return loss, student_output
