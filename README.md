# FluentNAS

FluentNAS is a modularized NAS Framework designed for One-Shot NAS.



## Overview

This repository includes the benchmark for NAS-Bench-101, NAS-Bench-201, NATS-Bench, Macro-NAS-Bench.

This repository support multiple NAS algorithms:

- Differentiable Architecture Search
- Single Path One Shot
- FairNAS
- AutoSlim
- AngleNAS

This repository support multiple dataset:

- CIFAR-10
- CIFAR-100
- ImageNet16-120



## Installation

Requirements:

- python>=3.7
- pytorch>=1.9



Intall from source code

```
git clone https://github.com/pprp/fluentnas.git
cd fluentnas
pip install -r requirements.txt
pip install -e .
```



## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
	--config configs/spos/spos_cifar10.py \
	--model_name OneShotNASBench201Network \
	--trainer_name NB201_Balance_Trainer \
	--log_name graduate_nb201_spos_sandwich-sampling \
	--dataset cifar10 --crit ce --lr 0.025
```



## Contributing

We welcome contributions to the library along with any potential issues or suggestions.



## Thanks

- MMRazor: https://github.com/open-mmlab/mmrazor
- SimMIM: https://github.com/microsoft/SimMIM
- NASLib: https://github.com/automl/NASLib
- MixPath: https://github.com/xiaomi-automl/MixPath
- Zerocostnas: https://github.com/SamsungLabs/zero-cost-nas
- RepDistiller: https://github.com/HobbitLong/RepDistiller
