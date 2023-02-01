import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(self):
        self.pairs = json.load(open('pairs_encoded.json'))
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
            
        return question, reply

    def __len__(self):
        return self.dataset_size
    