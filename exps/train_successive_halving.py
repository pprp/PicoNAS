# Change the main logic to Successive Halving Algorithm.

import os
import time

import torch
from args import get_args

import piconas.utils.utils as utils
from piconas.core import build_criterion, build_optimizer, build_scheduler
from piconas.datasets.build import build_dataloader
from piconas.models import build_model
from piconas.trainer import SuccessiveHalvingPyramid, build_trainer
from piconas.utils import set_random_seed


def main():
    cfg = get_args()

    # set envirs
    set_random_seed(cfg.seed, deterministic=True)

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

    train_dataloader = build_dataloader(
        type='train', dataset=cfg.dataset, config=cfg)

    val_dataloader = build_dataloader(
        type='val', dataset=cfg.dataset, config=cfg)

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

    SHPyramid = SuccessiveHalvingPyramid(trainer=trainer)

    SHPyramid.fit(train_loader=train_dataloader,
                  val_loader=val_dataloader, epoch=200)

    utils.time_record(start)


if __name__ == '__main__':
    main()
