# JAHS-Benchmark Test for Neural Architecture Search

## Introduction

evaluating the performance of NAS algorithms on the JAHSBench(NAS-Bench-201) dataset.

## Benchmark

https://github.com/automl/jahs_bench_201


1. Installation using pip

```
pip install jahs-bench
```

2. Download the source file 

```
python -m jahs_bench.download --target surrogates
```

## Usage

1. generate a config:

```
config = {
    'Optimizer': 'SGD',
    'LearningRate': 0.1,
    'WeightDecay': 5e-05,
    'Activation': 'Mish',
    'TrivialAugment': False,
    'Op1': 4,
    'Op2': 1,
    'Op3': 2,
    'Op4': 0,
    'Op5': 2,
    'Op6': 1,
    'N': 5,
    'W': 16,
    'Resolution': 1.0,
}
```

2. get the performance of the config:

```python
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", download=True)

# Query a random configuration
config = benchmark.sample_config()
results = benchmark(config, nepochs=200)

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # A dict
```


## Search space 

### Discrete search space

1. `activation` functions (Mish, Hardswish or ReLU)
2. `augmentations` (TrivialAugment or None)
3. `depth-multiplier` Each cell may be repeated `N` in {1, 3, 5}
4. `base-filters` The first convolution layer of any cell contains `W` in {4, 8, 16} filters, with the number of filters doubling in every subsequent convolution layer in the same cell.
5. `resolution-multiplier R` in in {0.25, 0.5, 1.0}.
6. `epoch` in {1, 2, ..., 200}.

### Continuous search space

7. `learning-rate` in [0.001, 1]
8. `weight-decay` in [1e-5, 1e-2]