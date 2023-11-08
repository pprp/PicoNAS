import json

import torch
from torch.utils.data import Dataset

from piconas.utils.get_dataset_api import NDS


class ZcDataset(Dataset):

    def __init__(self, search_space, json_path):
        # Load data from json file
        with open(json_path, 'r') as file:
            self.data = json.load(file)[search_space]
        self.nds_api = NDS(search_space)
        self.preprocess()

    def __len__(self):
        # Return the total number of samples in the data
        return 1000  # len(self.data)

    def preprocess(self):
        # get max length
        max_length = 0
        for uid in range(1000):
            sample = self.data[str(uid)]
            for key in sample:
                if max_length < len(sample[key]):
                    max_length = len(sample[key])

        # padding zero to max length
        for uid in range(1000):
            sample = self.data[str(uid)]
            for key in sample:
                sample[key] = sample[key] + [0] * (
                    max_length - len(sample[key]))

    def __getitem__(self, index):
        assert index < 1000, 'Index out of range'
        index = str(index)

        # Retrieve one sample from the dataset at the specified index
        sample = self.data[index]

        # Extract features (layerwise_zc) and label (gt) from the sample
        stack_list = []
        stack_list.extend(sample['synflow_layerwise'])
        stack_list.extend(sample['snip_layerwise'])
        stack_list.extend(sample['grad_norm_layerwise'])
        stack_list.extend(sample['fisher_layerwise'])
        features = torch.tensor(stack_list, dtype=torch.float)

        gt = self.nds_api.get_final_accuracy(int(index))
        label = torch.tensor(gt, dtype=torch.float)

        # Return the features and the label
        return features, label
