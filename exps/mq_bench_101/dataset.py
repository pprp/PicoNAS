import json

import torch
from torch.utils.data import Dataset


class ZcDataset(Dataset):
    def __init__(self, json_path):
        # Load data from json file
        with open(json_path, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        # Return the total number of samples in the data
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve one sample from the dataset at the specified index
        sample = self.data[index]

        # Extract features (layerwise_zc) and label (gt) from the sample
        features = torch.tensor(sample['layerwise_zc'], dtype=torch.float)
        label = torch.tensor(sample['gt'], dtype=torch.float)

        # Return the features and the label
        return features, label
