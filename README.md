# spos-cifar

single path one shot with CIFAR10

![](https://pic3.zhimg.com/v2-44d8ca5374bd7c45d345b75e8117e36a_b.jpg)


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
- [ ] NASBench201
- [ ] SPOS
- [ ] FairNAS
- [ ] AngleNAS
- [ ] BNNAS
- [ ] LandmarkReg
- [x] MacroBenchmark
- [x] SAM optimizer
- [ ] KD
