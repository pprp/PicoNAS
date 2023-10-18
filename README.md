# PicoNAS

PicoNAS is a modularized NAS Framework designed for One-Shot NAS. It is built on top of PyTorch and supports multiple NAS algorithms and datasets.

## Overview

This repository includes the benchmarks:

- NAS-Bench-101
- NAS-Bench-201
- NAS-Bench-301
- NATS-Bench
- Macro-NAS-Bench.

This repository support multiple NAS algorithms:

- Differentiable Architecture Search
- Single Path One Shot
- FairNAS
- AutoSlim
- AngleNAS
- [RD-NAS](https://github.com/pprp/PicoNAS/tree/dev/examples/rdnas)
- [MetaPool](https://github.com/pprp/PicoNAS/tree/dev/examples/meta_pool)

This repository support multiple dataset:

- CIFAR-10
- CIFAR-100
- ImageNet16-120

## Installation

Requirements:

- python>=3.7
- pytorch>=1.9
- nasbench

Note that tensorflow 1.x only support Python<=3.7 to install nasbench.

Intall from source code

```
git clone https://github.com/pprp/PicoNAS.git
cd PicoNAS
pip install -r requirements.txt
# install piconas
pip install -e .
```

## Introduction

PicoNAS is a modularized NAS Framework designed for One-Shot NAS. It contains the following components:
- Dataset: CIFAR-10, CIFAR-100, ImageNet16-120
- SuperNet: DARTS, SPOS, FairNAS, AutoSlim, AngleNAS
- Mutable: OneshotOP, DifferentiableOP, MutableOP
- Mutator: OneshotMutator, DifferentiableMutator, MutableMutator
- Evaluator: NB101Evaluator, NB201Evaluator, NB301Evaluator, NATSEvaluator, MacroNASEvaluator
- Trainer: NB101Trainer, NB201Trainer, NB301Trainer, NATSTrainer, MacroNASTrainer
- Predictor: SNIP, EPENAS, Fisher, GraSP, NWOT, ZEN, GradNorm, Synflow, TENAS, etc.


## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python exps/train.py \
  \
 --model_name OneShotNASBench201Network \
 --trainer_name NB201_Balance_Trainer \
 --log_name graduate_nb201_spos_sandwich-sampling \
 --dataset cifar10 --crit ce --lr 0.025
```


## Contributing

We welcome contributions to the library along with any potential issues or suggestions.

## Thanks

- MMRazor: <https://github.com/open-mmlab/mmrazor>
- SimMIM: <https://github.com/microsoft/SimMIM>
- NASLib: <https://github.com/automl/NASLib>
- MixPath: <https://github.com/xiaomi-automl/MixPath>
- Zerocostnas: <https://github.com/SamsungLabs/zero-cost-nas>
- RepDistiller: <https://github.com/HobbitLong/RepDistiller>
- ZenNAS: <https://github.com/idstcv/ZenNAS/>
- NAS-w-Proxy-data: <https://github.com/nabk89/NAS-with-Proxy-data>

## Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@misc{piconas2023,
  author = {Peijie Dong},
  title = {PicoNAS: A Modularized NAS Framework},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```

## TODO List

- [ ] update config system
- [ ] update log system
- [ ] update summarywriter
