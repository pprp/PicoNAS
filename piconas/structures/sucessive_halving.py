import random
from typing import Dict, List, Union

from piconas.core.losses import PairwiseRankLoss
from piconas.trainer.nb201_trainer import NB201Trainer


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

    def __len__(self) -> int:
        return len(self.data)

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
                 epoch_list: list = [1, 1, 2, 2],
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
        self.evaluator = trainer.evaluator
        self.K_init = K_init
        self.K_proposal = K_proposal
        self.epoch_list = epoch_list
        self.prior = prior

        self.pairwise_rankloss = PairwiseRankLoss()

        # init N levels
        self.pyramid = [Level() for _ in range(N)]
        # init first level
        for _ in range(K_init):
            current_brick = Brick(self.mutator.random_subnet)
            # zen score
            prior_score = self.evaluator.query_zerometric(
                current_brick.subnet_cfg)
            current_brick.prior_score = prior_score
            self.pyramid[0].append(current_brick)

    def __repr__(self) -> str:
        res = ''
        for i, level in enumerate(self.pyramid):
            res += f' => level-{i} len: {len(level)}'
            for brick in level:
                res += f' ==> {brick}'
        return res

    def perform(self, train_loader, val_loader):
        """From Level 1 to Level K:
            - Sort by prior scores.
            - Train top r * K arch in level-i for epoch_list[i]
            - Update Level info(num_iters, val_acc).
            - Upgrade top r * k arch to level-(i+1).
        """
        K = self.K_proposal
        for i in range(self.N):
            # In each Level
            # Get corresponding epochs
            epoch = self.epoch_list[i]

            # Get current level
            level = self.pyramid[i]

            # Sort by prior scores
            level.sort_by_prior()

            # Train top r * K
            for brick in level[:int(self.move_ratio * K)]:
                subnet_cfg = brick.subnet_cfg

                self.trainer.fit_specific(train_loader, val_loader, epoch,
                                          subnet_cfg)

                brick.num_iters += epoch
                brick.val_acc = self.trainer._validate(val_loader)

            # Upgrade top r * K
            if i + 1 < self.N:
                level.sort_by_val()

                for _ in range(int(self.move_ratio * K)):
                    brick = level.pop()
                    self.pyramid[i + 1].append(brick)
            K *= self.move_ratio

    def add_new_proposal(self):
        for _ in range(self.K_proposal):
            current_brick = Brick(self.mutator.random_subnet)
            # zen score
            prior_score = self.evaluator.query_zerometric(
                current_brick.subnet_cfg)
            current_brick.prior_score = prior_score
            self.pyramid[0].append(current_brick)

    def pairwise_ranking(self, train_loader):
        iter_loader = iter(train_loader)

        # sort by prior score
        for level in self.pyramid:
            level.sort_by_prior()

        # sample N-1 pair to calculate loss
        for i in range(self.N - 1):
            current_level = self.pyramid[i]
            next_level = self.pyramid[i + 1]

            current_brick = random.sample(
                current_level.data[:len(current_level) // 2], 1)[0]
            next_brick = random.sample(next_level.data[:len(next_level) // 2],
                                       1)[0]
            inputs, labels = iter_loader.next()
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Process the current brick
            current_subnet = current_brick.subnet_cfg
            self.trainer.mutator.set_subnet(current_subnet)
            output = self.trainer.model(inputs)
            loss1 = self.trainer._compute_loss(output, labels)
            prior1 = current_brick.prior_score

            # Process the next brick
            next_subnet = next_brick.subnet_cfg
            self.trainer.mutator.set_subnet(next_subnet)
            output = self.trainer.model(inputs)
            loss2 = self.trainer._compute_loss(output, labels)
            prior2 = next_brick.prior_score

            # Calculate Pairwise Rank loss
            loss3 = 0.5 * self.pairwise_rankloss(prior1, prior2, loss1, loss2)

            # Sum the losses
            sum_loss = loss1 + loss2 + loss3
            sum_loss.backward()

    def fit(self, train_loader, val_loader, epoch):
        """Main Search Process.

        In each loop:
            Perform SH Pyramid process.
            Using Pairwise Ranking Loss.
            Add new Proposal.
        """
        # M = self.K_init
        # Tranverse for 10 times.
        for _ in range(epoch):
            self.perform(train_loader, val_loader)
            self.pairwise_ranking(train_loader)
            self.add_new_proposal()
