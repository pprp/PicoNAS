import numpy as np
import torchvision.transforms as transforms


class SimMIMTransform:

    def __init__(self):
        self.transform_img = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.mask_generator = MaskGenerator(
            input_size=32,
            mask_patch_size=2,
            model_patch_size=2,  # for vit
            mask_ratio=0.6,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        return img, mask


class MaskGenerator:
    """Generate mask with ratio"""

    def __init__(self,
                 input_size=32,
                 mask_patch_size=2,
                 model_patch_size=2,
                 mask_ratio=0.6):

        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask
