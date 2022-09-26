import argparse

import torch
import torch.nn as nn

from pplib.datasets.data_simmim import build_loader_simmim
from pplib.evaluator import NATSEvaluator
from pplib.models.nats.nats_supernet import MAESupernetNATS
from pplib.trainer.nats_trainer import MAENATSTrainer
from pplib.utils.loggings import get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser('rank evaluation')
    parser.add_argument(
        '--bench_path',
        type=str,
        default='./data/benchmark/nats_cifar10_acc_rank.yaml',
        help='benchmark file path',
    )

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='checkpoints/nats_mae_cosine/xx.pth.tar',
        help='path of supernet checkpoint.',
    )

    parser.add_argument(
        '--num_sample',
        type=int,
        default=10,
        help='number of sample for rank evaluation.',
    )

    args = parser.parse_args()

    logger = get_logger('eval_mae')

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # generate supernet model
    supernet = MAESupernetNATS('cifar10')

    # criterion
    criterion = nn.MSELoss().to(device)

    # load supernet checkpoints
    state = torch.load(args.ckpt_path)['state_dict']
    supernet.load_state_dict(state, strict=False)

    # build valid dataloader
    dataloader = build_loader_simmim(is_train=False)

    # get trainer
    trainer = MAENATSTrainer(
        supernet, mutator=None, device=device, criterion=criterion)

    evaluator = NATSEvaluator(
        trainer,
        dataloader,
        bench_path=args.bench_path,
        num_sample=args.num_sample)

    evaluator.compute_rank_consistency()
