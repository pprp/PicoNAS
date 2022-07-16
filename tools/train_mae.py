import argparse
import math
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pplib.datasets import build_dataloader
from pplib.models.mae.mae_model import MAE_ViT, ViT_Classifier
from pplib.utils.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument(
        '--output_model_path',
        type=str,
        default='vit-t-classifier-from_scratch.pt')

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_loader = build_dataloader(name='cifar10', type='train', config=cfg)
    val_loader = build_dataloader(name='cifar10', type='val', config=cfg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        writer = SummaryWriter(
            os.path.join('logdir', 'cifar10', 'pretrain-cls'))
    else:
        model = MAE_ViT()
        writer = SummaryWriter(
            os.path.join('logdir', 'cifar10', 'scratch-cls'))
    model = ViT_Classifier(model.encoder, num_classes=10).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    def acc_fn(logit, label):
        return torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_learning_rate * args.batch_size / 256,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay)

    def lr_func(epoch):
        return min((epoch + 1) / (args.warmup_epoch + 1e-8),
                   0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0.
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        access = []
        for img, label in tqdm(iter(train_loader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            access.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(access) / len(access)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, \
                average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            access = []
            for img, label in tqdm(iter(val_loader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                access.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(access) / len(access)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, \
                    average validation acc is {avg_val_acc}.')

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')
            torch.save(model, args.output_model_path)

        writer.add_scalar(
            'cls/loss', {
                'train': avg_train_loss,
                'val': avg_val_loss
            },
            global_step=e)
        writer.add_scalar(
            'cls/acc', {
                'train': avg_train_acc,
                'val': avg_val_acc
            },
            global_step=e)
