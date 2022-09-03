import random
import time
from typing import Dict, List, Union

import torch
from mmcv.cnn import get_model_complexity_info

import pplib.utils.utils as utils
from pplib.core.losses import PairwiseRankLoss
from pplib.evaluator.nb201_evaluator import NB201Evaluator
from pplib.models.nasbench201 import OneShotNASBench201Network
from pplib.nas.mutators import OneShotMutator
from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer
from .registry import register_trainer


@register_trainer
class NB201Trainer(BaseTrainer):
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
        model: OneShotNASBench201Network,
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
            searching=searching,
        )

        # init flops
        self._init_flops()

        if self.mutator is None:
            # use alias to build search group
            self.mutator = OneShotMutator(with_alias=True)
            self.mutator.prepare_from_supernet(model)

        # evaluate the rank consistency
        self.evaluator = None

        # pairwise rank loss
        self.pairwise_rankloss = PairwiseRankLoss()

        # record current rand_subnet
        self.rand_subnet = None

    def _build_evaluator(self, dataloader, num_sample=50):
        self.evaluator = NB201Evaluator(self, dataloader, num_sample)

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

            # FairNAS
            # loss, outputs = self._forward_fairnas(batch_inputs)

            # Single Path One Shot
            loss, outputs = self.forward(batch_inputs, mode='loss')
            loss.backward()

            # SPOS with pairwise rankloss
            # loss, outputs = self._forward_pairwise_loss(batch_inputs)

            # spos with pairwise rankloss + cc distill
            # loss, outputs = self._forward_pairwise_loss_with_distill(
            #     batch_inputs)

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
                    f'Step: {step:03} Train loss: {loss.item():.4f} Top1 acc: {top1_tacc.avg:.3f} Top5 acc: {top5_tacc.avg:.3f} Current geno: {self.evaluator.generate_genotype(self.rand_subnet, self.mutator)}'
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
        if self.searching:
            self.rand_subnet = self.mutator.random_subnet
            self.mutator.set_subnet(self.rand_subnet)
        return self.model(inputs)

    def _predict(self, batch_inputs, subnet_dict: Dict = None):
        """Network forward step. Low Level API"""
        inputs, labels = batch_inputs
        inputs = self._to_device(inputs, self.device)
        labels = self._to_device(labels, self.device)
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
                f' Top5 acc: {top5_vacc.avg} Current geno: {self.evaluator.generate_genotype(self.rand_subnet, self.mutator)}'
            )
        return val_loss / (step + 1), top1_vacc.avg, top5_vacc.avg

    def fit(self, train_loader, val_loader, epochs):
        """Fits. High Level API
        Fit the model using the given loaders for the given number
        of epochs.
        """
        # build evaluator
        if self.evaluator is None:
            self._build_evaluator(val_loader, num_sample=50)

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
                assert self.evaluator is not None
                kt, ps, sp = self.evaluator.compute_rank_consistency(
                    self.mutator)
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

        # sample the first subnet
        self.rand_subnet = self.mutator.random_subnet
        self.mutator.set_subnet(self.rand_subnet)
        outputs = self.model(inputs)
        loss1 = self._compute_loss(outputs, labels)
        loss1.backward()
        flops1 = self.get_subnet_flops(self.rand_subnet)

        # sample the second subnet
        self.rand_subnet = self.mutator.random_subnet
        self.mutator.set_subnet(self.rand_subnet)
        outputs = self.model(inputs)
        loss2 = self._compute_loss(outputs, labels)
        loss2.backward(retain_graph=True)
        flops2 = self.get_subnet_flops(self.rand_subnet)

        # pairwise rank loss
        # lambda settings:
        #       1. min(2, self.current_epoch/10.)
        #       2. 2 * np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs)

        loss3 = self._lambda * self.pairwise_rankloss(flops1, flops2, loss1,
                                                      loss2)
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


class Brick:
    """Basic element to store the information.

    Args:
        subnet_cfg (_type_): _description_
        num_iters (int, optional): _description_. Defaults to 0.
        prior_score (int, optional): _description_. Defaults to 0.
    """

    def __init__(self, subnet_cfg, num_iters=0, prior_score=0, val_acc=0):

        self._subnet_cfg = subnet_cfg
        self._num_iters = num_iters
        self._prior_score = prior_score
        self._val_acc = val_acc

    @property
    def subnet_cfg(self) -> Dict:
        return self._subnet_cfg

    @subnet_cfg.setter
    def subnet_cfg(self, subnet_cfg: Dict) -> None:
        self._subnet_cfg = subnet_cfg

    @property
    def num_iters(self) -> int:
        return self._num_iters

    @num_iters.setter
    def num_iters(self, num_iters) -> None:
        self._num_iters = num_iters

    @property
    def prior_score(self) -> float:
        return self._prior_score

    @prior_score.setter
    def prior_score(self, prior_score) -> None:
        self._prior_score = prior_score

    @property
    def val_acc(self):
        return self._val_acc

    @val_acc.setter
    def val_acc(self, val_acc: float):
        self._val_acc = val_acc

    def __repr__(self):
        str = 'Brick:'
        str += f' => subnet: {self.subnet_cfg} '
        str += f'num_iters: {self.num_iters} '
        str += f'prior: {self.prior_score} '
        str += f'val_acc: {self.val_acc} '
        return str


class Level(list):
    """Basic element in SHPyramid.

    self.data = [Brick1, Brick2, ..., BrickN]
    """

    def __init__(self, initdata: Union[Brick, List] = None):
        self.data: List[Brick] = []
        if initdata is not None:
            if isinstance(initdata, list):
                self.data = [initdata]
            elif isinstance(initdata, Brick):
                self.data.append(initdata)
            else:
                raise NotImplementedError

    def pop(self, index=0):
        return self.data.pop(index)

    @property
    def subnet_cfg(self) -> Dict:
        return [item.subnet_cfg for item in self.data]

    @property
    def num_iters(self) -> int:
        return [item.num_iters for item in self.data]

    @property
    def prior_score(self) -> float:
        return [item.prior_score for item in self.data]

    @property
    def val_acc(self) -> float:
        return [item.val_acc for item in self.data]

    def append(self, item: Brick) -> None:
        self.data.append(item)

    def extend(self, item: List) -> None:
        self.data.extend(item)

    def set_iters(self, subnet: Dict, num_iters: int) -> None:
        assert subnet in self.subnet_cfg
        for item in self.data:
            if subnet == item.subnet_cfg:
                item.num_iters = num_iters

    def set_prior_score(self, subnet: Dict, prior_score: float) -> None:
        assert subnet in self.subnet_cfg
        for item in self.data:
            if subnet == item.subnet_cfg:
                item.prior_score = prior_score

    def sort_by_iters(self) -> None:
        self.data = sorted(
            self.data, key=lambda brick: brick.num_iters, reverse=True)

    def sort_by_prior(self) -> None:
        self.data = sorted(
            self.data, key=lambda brick: brick.prior_score, reverse=True)

    def sort_by_val(self) -> None:
        self.data = sorted(
            self.data, key=lambda brick: brick.val_acc, reverse=True)

    def __repr__(self):
        res = 'Level: \n'
        for item in self.data:
            res += f' => subnet {item.subnet_cfg} '
            res += f'iters: {item.num_iters} '
            res += f'prior_score: {item.prior_score} '
            res += f'val_acc: {item.val_acc} \n'
        return res


class SuccessiveHalvingPyramid:
    """Sucessive Halving Pyramid Pool.

    Args:
        N (int, optional): Total number of Level. Defaults to 4.
        r (float, optional): Move Ratio. Defaults to 0.5.
        K_init (int, optional): Number of sampled config in the
            initialization round.
        K_proposal (int, optional): Proposal size in the afterwards
            round.
    """

    def __init__(self,
                 N: int = 4,
                 r: float = 0.5,
                 epoch_list: list = [1, 2, 3, 12],
                 trainer: NB201Trainer = None,
                 K_init: int = 16 * 3,
                 K_proposal: int = 10 * 3,
                 prior: str = 'flops') -> None:
        super().__init__()
        assert trainer is not None
        assert len(epoch_list) == N
        assert prior in ['flops', 'zen']

        self.N = N
        self.move_ratio = r
        self.trainer = trainer
        self.mutator = trainer.mutator
        self.K_init = K_init
        self.K_proposal = K_proposal
        self.epoch_list = epoch_list

        # init N levels
        self.pyramid = [Level() for _ in range(N)]
        # init first level
        for _ in range(K_init):
            current_brick = Brick(self.mutator.random_subnet)
            self.pyramid[0].append(current_brick)

    def perform(self, train_loader, val_loader):
        """From Level 1 to Level K:
            - Sort by prior scores.
            - Train top r * K arch in level-i for epoch_list[i]
            - Update Level info(num_iters, val_acc).
            - Upgrade top r * k arch to level-(i+1).
        """

        for i in range(self.N):
            # In each Level
            # Get corresponding epochs
            epoch = self.epoch_list[i]
