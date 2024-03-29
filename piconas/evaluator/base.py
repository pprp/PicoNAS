from abc import abstractmethod


class Evaluator:
    def __init__(self, trainer, bench_path=None, dataset: str = 'cifar10'):
        self.trainer = trainer
        self.bench_path = bench_path
        self.dataset = dataset

    @abstractmethod
    def load_benchmark(self):
        """load benchmark to get dict."""
        pass

    def compute_rank_consistency(self):
        """compute rank consistency."""
        pass

    def sample_archs(self, bench_dict):
        """sample from benchmark."""
