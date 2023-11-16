from typing import Dict, List

import torch


class ViTBenchAPI:
    def __init__(self, api_path: str):
        self.data: List[Dict] = self.load_json(api_path)

    def load_json(self, api_path: str) -> List[Dict]:
        return torch.load(api_path)

    def query_by_idx(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self.data):
            raise IndexError(
                f'Index out of range. Max index: {len(self.data) - 1}')
        return self.data[idx]

    def query_by_arch(self, arch: Dict) -> List[Dict]:
        results = []
        for item in self.data:
            if item['arch'] == arch:
                results.append(item)
        return results

    def query_by_cifar100_base(self, base_acc: float) -> List[Dict]:
        results = []
        for item in self.data:
            if (
                'cifar100' in item
                and 'base' in item['cifar100']
                and item['cifar100']['base'] == base_acc
            ):
                results.append(item)
        return results

    def query_by_flowers_base(self, base_acc: float) -> List[Dict]:
        results = []
        for item in self.data:
            if (
                'flowers' in item
                and 'base' in item['flowers']
                and item['flowers']['base'] == base_acc
            ):
                results.append(item)
        return results

    def query_by_chaoyang_base(self, base_acc: float) -> List[Dict]:
        results = []
        for item in self.data:
            if (
                'chaoyang' in item
                and 'base' in item['chaoyang']
                and item['chaoyang']['base'] == base_acc
            ):
                results.append(item)
        return results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.query_by_idx(idx)


# import torch

# api = ViTBenchAPI('checkpoints/af_100.pth')

# # Query by index
# result = api.query_by_idx(0)
# print(result)

# # Query by architecture
# arch = {
#     'hidden_dim': 192,
#     'mlp_ratio': [3.5, 4.0, 3.5, 4.0, 3.5, 3.5, 4.0, 3.5, 4.0, 4.0, 4.0, 3.5],
#     'depth': 12,
#     'num_heads': [4, 4, 3, 4, 3, 3, 4, 4, 3, 3, 3, 4]
# }
# results = api.query_by_arch(arch)
# print(results)

# # Query by CIFAR-100 base accuracy
# base_acc = 68.66
# results = api.query_by_cifar100_base(base_acc)
# print(results)

# # Query by Flowers base accuracy
# base_acc = 53.6835
# results = api.query_by_flowers_base(base_acc)
# print(results)

# # Query by Chaoyang base accuracy
# base_acc = 81.9542
# results = api.query_by_chaoyang_base(base_acc)
# print(results)
