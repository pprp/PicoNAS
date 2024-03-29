import argparse
import os
import time

import torch

import piconas.utils.utils as utils
from piconas.core import build_criterion, build_optimizer, build_scheduler

# from piconas.datasets.build import build_dataloader
from piconas.evaluator import MacroEvaluator
from piconas.models import build_model
from piconas.trainer import build_trainer


def get_args():
    parser = argparse.ArgumentParser('train macro benchmark')
    parser.add_argument(
        '--work_dir', type=str, default='./work_dir', help='experiment name'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data/cifar', help='path to the dataset'
    )

    parser.add_argument(
        '--model_name', type=str, default='MAESupernetNATS', help='name of model'
    )
    parser.add_argument(
        '--trainer_name', type=str, default='NATSMAETrainer', help='name of trainer'
    )
    parser.add_argument(
        '--log_name',
        type=str,
        default='NATSMAETrainer',
        help='name of this experiments',
    )

    parser.add_argument('--crit', type=str, default='mse',
                        help='decide the criterion')
    parser.add_argument(
        '--optims', type=str, default='sgd', help='decide the optimizer'
    )
    parser.add_argument(
        '--sched', type=str, default='cosine', help='decide the scheduler'
    )

    parser.add_argument('--classes', type=int, default=10,
                        help='dataset classes')
    parser.add_argument('--layers', type=int, default=20, help='batch size')
    parser.add_argument(
        '--num_choices', type=int, default=4, help='number choices per layer'
    )
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='batch size')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument(
        '--val_interval', type=int, default=5, help='validate and save frequency'
    )
    parser.add_argument(
        '--random_search', type=int, default=1000, help='validate and save frequency'
    )
    # ******************************* dataset *******************************#
    parser.add_argument(
        '--dataset', type=str, default='simmim', help='path to the dataset'
    )
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument('--cutout_length', type=int,
                        default=16, help='cutout length')
    parser.add_argument(
        '--auto_aug', action='store_true', default=False, help='use auto augmentation'
    )
    parser.add_argument(
        '--resize', action='store_true', default=False, help='use resize'
    )
    args = parser.parse_args()
    return args


def main():
    cfg = get_args()

    print(cfg)

    # dump config files
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.trainer_name)
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
    current_exp_name = f'{cfg.model_name}-{cfg.trainer_name}-{cfg.log_name}.yaml'
    cfg.dump(os.path.join(cfg.work_dir, current_exp_name))

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # train_dataloader = build_dataloader(
    #     type='train', dataset=cfg.dataset, config=cfg)

    # val_dataloader = build_dataloader(
    #     type='val', dataset=cfg.dataset, config=cfg)

    model = build_model(cfg.model_name)

    criterion = build_criterion(cfg.crit).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer)

    model = model.to(device)

    trainer = build_trainer(
        cfg.trainer_name,
        model=model,
        mutator=None,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        searching=True,
        device=device,
        log_name=cfg.log_name,
    )

    start = time.time()

    num_samples = [20, 50, 100]

    for num_sample in num_samples:
        evaluator = MacroEvaluator(
            trainer=trainer,
            dataset='cifar10',
            num_sample=num_sample,
        )
        kt, ps, sp = evaluator.compute_rank_by_flops()

    utils.time_record(start)


if __name__ == '__main__':
    main()
