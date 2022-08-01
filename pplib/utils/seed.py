# From mmcv
import random

import numpy as np
import torch


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    """Function to initialize each worker.
    The seed of each worker equals to
    ``num_worker * rank + worker_id + user_seed``.
    Args:
        worker_id (int): Id for each worker.
        num_workers (int): Number of workers.
        rank (int): Rank in distributed training.
        seed (int): Random seed.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set ``torch.backends.cudnn.deterministic``
            to True and ``torch.backends.cudnn.benchmark`` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
