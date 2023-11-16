import torch
import torch.nn as nn

from piconas.datasets.predictor.nb201_dataset import Nb201DatasetPINAT


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=294,  # layerwise zc
        sequence_length=256,
        patch_size=16,
        dim=512,
        depth=4,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.1,
    ):
        super(MLP, self).__init__()
        self.project = nn.Linear(input_dim, dim)

        self.mlp_blocks = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(depth)])
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, input):
        x = input['zcp_layerwise']
        x = self.project(x)
        for mixer_block in self.mlp_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.fc(x).view(-1)
        return x


if __name__ == '__main__':
    model = MLP()

    # Assuming Nb201DatasetPINAT is defined elsewhere
    test_set = Nb201DatasetPINAT(
        split='all', data_type='test', data_set='cifar10')

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=0, drop_last=False
    )

    for batch in loader:
        input = batch
        score = model(input)
        print(score)
