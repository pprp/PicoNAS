class Evaluator:

    def __init__(self, trainer, dataloader, bench_path):
        self.trainer = trainer
        self.dataloader = dataloader
        self.bench_path = bench_path

    def load_benchmark(self):
        """load benchmark to get dict."""
        pass

    def compute_rank_consistency(self):
        """compute rank consistency."""
        pass

    def sample_archs(self, bench_dict):
        """sample from benchmark."""