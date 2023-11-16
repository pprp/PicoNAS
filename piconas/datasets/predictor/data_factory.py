from torch.utils.data import DataLoader

from piconas.datasets.predictor.nb101_dataset import Nb101DatasetPINAT
from piconas.datasets.predictor.nb201_dataset import Nb201DatasetPINAT


def create_dataloader(args):
    # load dataset
    if args.bench == '101':
        train_set = Nb101DatasetPINAT(split=args.train_split, data_type='train')
        test_set = Nb101DatasetPINAT(split=args.eval_split, data_type='test')
    elif args.bench == '201':
        train_set = Nb201DatasetPINAT(
            split=int(args.train_split), data_type='train', data_set=args.dataset
        )
        test_set = Nb201DatasetPINAT(
            split='all', data_type='test', data_set=args.dataset
        )
    else:
        raise ValueError('No defined NAS bench!')

    # initialize dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0 if args.eval_split == '100' else 16,
        drop_last=True,
    )
    return train_loader, test_loader, train_set, test_set
