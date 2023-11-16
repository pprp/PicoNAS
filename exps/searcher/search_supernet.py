import argparse
import time

import torch
import torch.nn as nn
from thop import profile

# from torchsummary import summary
from tqdm import tqdm

import piconas.utils.utils as utils
from piconas.datasets import build_dataloader
from piconas.models import SearchableMobileNet


def get_args():
    parser = argparse.ArgumentParser('Single_Path_One_Shot')
    parser.add_argument(
        '--exp_name',
        type=str,
        default='spos_cifar10',
        required=True,
        help='experiment name',
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data/', help='path to the dataset'
    )
    parser.add_argument('--classes', type=int, default=10,
                        help='dataset classes')
    parser.add_argument('--layers', type=int, default=20, help='batch size')
    parser.add_argument(
        '--num_choices', type=int, default=4, help='number choices per layer'
    )
    parser.add_argument('--batch_size', type=int,
                        default=96, help='batch size')
    parser.add_argument('--epochs', type=int, default=600, help='batch size')
    parser.add_argument(
        '--learning_rate', type=float, default=0.025, help='initial learning rate'
    )
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float,
                        default=3e-4, help='weight decay')
    parser.add_argument(
        '--val_interval', type=int, default=5, help='validate and save frequency'
    )
    parser.add_argument(
        '--random_search', type=int, default=1000, help='validate and save frequency'
    )
    # ******************************* dataset *******************************#
    parser.add_argument(
        '--dataset', type=str, default='cifar10', help='path to the dataset'
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
    print(args)
    return args


def train(
    args, epoch, train_data, device, model, criterion, optimizer, scheduler, supernet
):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    train_data = tqdm(train_data)
    train_data.set_description(
        '[%s%04d/%04d %s%f]'
        % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_lr()[0])
    )
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if supernet:
            choice = utils.random_choice(args.num_choices, args.layers)
            outputs = model(inputs, choice)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # if args.dataset == 'cifar10':
        loss.backward()
        # elif args.dataset == 'imagenet':
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {
            'train_loss': '%.6f' % (train_loss / (step + 1)),
            'train_acc': '%.6f' % top1.avg,
        }
        train_data.set_postfix(log=postfix)


def validate(args, epoch, val_data, device, model, criterion, supernet, choice=None):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            if supernet:
                if choice is None:
                    choice = utils.random_choice(args.num_choices, args.layers)
                outputs = model(inputs, choice)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
        print(
            '[Val_Accuracy epoch:%d] val_loss:%f, val_acc:%f'
            % (epoch + 1, val_loss / (step + 1), val_top1.avg)
        )
        return val_top1.avg


def main():
    # args & device
    args = get_args()
    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # dataset
    assert args.dataset in ['cifar10', 'imagenet']
    train_loader = build_dataloader(name='cifar10', type='train', config=args)
    val_loader = build_dataloader(name='cifar10', type='val', config=args)
    # SinglePath_OneShot
    model = SearchableMobileNet()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), args.learning_rate, args.momentum, args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - (epoch / args.epochs)
    )

    # flops & params & structure
    flops, params = profile(
        model,
        inputs=(torch.randn(1, 3, 32, 32),)
        if args.dataset == 'cifar10'
        else (torch.randn(1, 3, 224, 224),),
        verbose=False,
    )
    # print(model)
    print(
        'Random Path of the Supernet: Params: %.2fM, Flops:%.2fM'
        % ((params / 1e6), (flops / 1e6))
    )
    model = model.to(device)
    # summary(model, (3, 32, 32) if args.dataset == 'cifar10'
    # else (3, 224, 224))

    # train supernet
    start = time.time()
    for epoch in range(args.epochs):
        train(
            args,
            epoch,
            train_loader,
            device,
            model,
            criterion,
            optimizer,
            scheduler,
            supernet=True,
        )
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            validate(args, epoch, val_loader, device,
                     model, criterion, supernet=True)

            utils.save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                },
                'search_supernet',
                epoch + 1,
                tag=f'{args.exp_name}_super',
            )

    utils.time_record(start)


if __name__ == '__main__':
    main()
