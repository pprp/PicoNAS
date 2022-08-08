import unittest
from unittest import TestCase

import numpy as np
from einops import rearrange

from pplib.datasets import build_dataloader, build_dataset
from pplib.utils.config import Config
from pplib.utils.logging import get_logger


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattribute__ = dict.__getitem__


class TestDataset(TestCase):

    def test_dataset(self):
        dataset = build_dataset(type='train', dataset='cifar10')
        assert dataset is not None
        for i, (img, label) in enumerate(dataset):
            if i > 2:
                break
            # print(img.shape, label)

    def test_dataloader(self):
        dataloader = build_dataloader(type='train', dataset='cifar10')
        assert dataloader is not None

        for i, (img, label) in enumerate(dataloader):
            if i > 2:
                break
            # print(img.shape, label.shape)

    def convert2pltimg(self, img):
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        return img

    def test_dataloader_simmim(self):
        loader = build_dataloader(type='val', dataset='simmim')

        # import ipdb; ipdb.set_trace()
        for i, (img, mask, _) in enumerate(loader):
            if i > 2:
                break

            import matplotlib.pyplot as plt

            print(img.shape, mask.shape)

            plt.imshow(self.convert2pltimg(img[0]))
            plt.savefig('origin.png')

            # 2, 3, 32, 32 -> 2, 16 * 16, 2*2*3
            img = rearrange(
                img, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=16, p2=16)

            # mask [2, 16, 16] -> [2, 16 * 16]
            mask = rearrange(mask, 'b h w -> b (h w)')

            mask = mask.unsqueeze(-1).repeat(1, 1, 12)

            # shape: [2, 16*16, 2*2*3]
            masked_img = mask * img

            # shape: [2, 16*16, 2*2*3] -> [2, 3, 2 * 16, 2 * 16]
            masked_img = rearrange(
                masked_img,
                'b (p1 p2) (c h w) -> b c (p1 h) (p2 w)',
                p1=16,
                p2=16,
                c=3,
                h=2,
                w=2,
            )

            # shape: [2, 3, 32, 32] -> [2, 32, 32, 3]
            # masked_img = rearrange(masked_img, 'b c h w -> b h w c')

            plt.imshow(self.convert2pltimg(masked_img[0]))
            plt.savefig('./test_masked_img.png')

            # mask_token = nn.Parameter(torch.zeros(1, 1, C))
            # mask_token = mask_token.expand(B, L, -1) # [2 1024 3]

            # mask: [2, 16, 16] => mask: [2, 256] => mask: [2, 256, 1]
            # w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)

            # img: [2, 1024, 3]
            # w: [2, 256, 1]
            # mask_token: [2, 1024, 3]
            # img = img * (1 - w) + mask_token * w


if __name__ == '__main__':
    unittest.main()
