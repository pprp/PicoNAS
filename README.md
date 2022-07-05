# pplib

single path one shot with CIFAR10


## MAE

- 1. pretrain:

```bash
python tools/mae_pretrain.py
```

- 2. retrain

```bash
python tools/train_classifier.py
```

## SPOS

- Search Space 1: Official ShuffleNet-based Search space.

```python
python tools/train_spos.py
```

- Search Space 2: Macro BenchMark-based Search space.

```python
python tools/train_macro.py
```

## Rank

To evaluate the rank consistency in nas, run:

```python
python tools/eval_rank.py
```


## TODO LIST

- [ ] SIMMIM + SupMIM
- [x] NASBench201
- [x] SPOS
- [ ] FairNAS
- [x] AngleNAS
- [ ] BNNAS
- [ ] LandmarkReg
- [x] MacroBenchmark
- [x] SAM optimizer
- [ ] KD


## Thanks

- MMRazor: https://github.com/open-mmlab/mmrazor
- SimMIM: https://github.com/microsoft/SimMIM
- NASLib: https://github.com/automl/NASLib


## NASLib 

runner: 属于入口，兼容各种用法

- api：(naslib/utils/get_dataset_api) 获取api，得到json的访问
- predictor: 预测器，根据不同type的方法如zen，snip等进行计算，提供查询功能，得到对应分数
- search_space: Graph类
- evaluator: 
