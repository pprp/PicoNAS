from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from pplib.nas.mutators import OneShotMutator
from pplib.utils.utils import AvgrageMeter, accuracy
from .base import BaseTrainer


class SPOSTrainer(BaseTrainer):

    def __init__(
        self,
        model: nn.Module,
        mutator: OneShotMutator,
        dataloader: Dict,
        optimizer,
        criterion,
        scheduler,
        epochs: int,
        searching: bool = True,
        num_choices: int = 4,
        num_layers: int = 20,
        device: torch.device = None,
    ):

        self.epochs = epochs
        self.model = model
        self.searching = searching
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.num_choices = num_choices
        self.num_layers = num_layers
        self.mutator = mutator

    def train(self, epoch: int):
        self.model.train()
        train_loss = 0.0
        top1 = AvgrageMeter()
        train_dataloader = tqdm(self.dataloader['train'])
        train_dataloader.set_description(
            '[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, self.epochs, 'lr:',
                                    self.scheduler.get_lr()[0]))
        for step, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            if self.searching:
                rand_subnet = self.mutator.random_subnet
                self.mutator.set_subnet(rand_subnet)
                outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            self.optimizer.step()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {
                'train_loss': '%.6f' % (train_loss / (step + 1)),
                'train_acc': '%.6f' % top1.avg
            }
            train_dataloader.set_postfix(log=postfix)

    def valid(self, epoch, subnet_dict: Dict = None):
        self.model.eval()
        val_loss = 0.0
        val_top1 = AvgrageMeter()
        val_dataloader = self.dataloader['val']
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(val_dataloader):
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                if self.searching:
                    # during searching phase, test random subnet
                    rand_subnet = self.mutator.random_subnet
                    self.mutator.set_subnet(rand_subnet)
                    outputs = self.model(inputs)
                else:
                    # during evaluation phase, test specific subnet
                    self.mutator.set_subnet(subnet_dict)
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                n = inputs.size(0)
                val_top1.update(prec1.item(), n)
            print('[Val_Accuracy epoch:%d] val_loss:%f, val_acc:%f' %
                  (epoch + 1, val_loss / (step + 1), val_top1.avg))
            return val_top1.avg
