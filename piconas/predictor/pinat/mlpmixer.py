import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from piconas.datasets.predictor.nb201_dataset import Nb201DatasetPINAT


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
    )


class MLPMixer(nn.Module):
    def __init__(
        self,
        input_dim=294,  # layerwise zc
        sequence_length=256,
        patch_size=16,
        dim=512,
        depth=4,
        emb_out_dim=1,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.1,
    ):
        super(MLPMixer, self).__init__()
        self.project = nn.Linear(input_dim, dim)

        self.mixer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout)),
                    PreNormResidual(
                        dim, FeedForward(dim, expansion_factor_token, dropout)
                    ),
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, emb_out_dim)

    def forward(self, input):
        x = input['zcp_layerwise']
        x = self.project(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.fc(x).view(-1)  # Changed from self.head to self.fc
        return x


if __name__ == '__main__':
    model = MLPMixer()

    # Assuming Nb201DatasetPINAT is defined elsewhere
    test_set = Nb201DatasetPINAT(split='all', data_type='test', data_set='cifar10')

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=0, drop_last=False
    )

    for batch in loader:
        input = batch
        score = model(input)
        print(score)
