# NASBench101

This is a PyTorch implementation of the NASBench101 model. The model is based on the paper [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635).

official code: https://github.com/google-research/nasbench

## Requirements

Download official checkpoints, here we use the `nasbench_only108.tfrecord` file.

Subset of the dataset with only models trained at 108 epochs:

https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

Size: ~499 MB, SHA256: 4c39c3936e36a85269881d659e44e61a245babcb72cb374eacacf75d0e5f4fd1

We thank the authors for providing the PyTorch vision of NASBench: https://github.com/romulus0914/NASBench-PyTorch

## Usage

Get the PyTorch architecture of a network like this:

```python
from nasbench_pytorch.model import Network as NBNetwork
from nasbench import api


nasbench_path = '$path_to_downloaded_nasbench'
nb = api.NASBench(nasbench_path)

net_hash = '$some_hash'  # you can get hashes using nasbench.hash_iterator()
m = nb.get_metrics_from_hash(net_hash)
ops = m[0]['module_operations']
adjacency = m[0]['module_adjacency']

net = NBNetwork((adjacency, ops))
```

## Disclaimer

Modified from NASBench: A Neural Architecture Search Dataset and Benchmark. graph_util.py and model_spec.py are directly copied from the original repo. Original license can be found here.

Please note that this repo is only used to train one possible architecture in the search space, not to generate all possible graphs and train them.
