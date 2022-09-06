import torch

from pplib.trainer import NB201Trainer
from pplib.utils.logging import get_logger


class RandomSearcher(object):

    def __init__(self, trainer: NB201Trainer = None, model_path=None):
        self.trainer = trainer
        self.recorder = []
        self.logger = get_logger(name='RandomSearcher')

        self.model = self.trainer.model
        if model_path is None:
            model_path = './checkpoints/test_nb201_spos/test_nb201_spos_macro_ckpt_0191.pth.tar'
        state_dict = torch.load(model_path)['state_dict']
        self.model.load_state_dict(state_dict)

    def search(self,
               epochs=20,
               sample_per_epoch=3,
               train_loader=None,
               val_loader=None):
        best_top1_err = 100
        best_subnet = None
        for e in range(epochs):
            self.logger.info(f'Searching in Epoch {e}')
            tmp_recorder = []
            for s in range(sample_per_epoch):
                subnet = self.trainer.mutator.random_subnet
                self.trainer.mutator.set_subnet(subnet)
                top1_err, top5_err = self.trainer.get_subnet_error(
                    subnet, train_loader, val_loader)
                tmp_recorder.append(top1_err)
                if top1_err < best_top1_err:
                    best_top1_err = top1_err
                    best_subnet = subnet
                self.logger.info(
                    f'The top1 error of the {s}-th subnet: {top1_err}')
            self.recorder.append(tmp_recorder)

        genotype = self.trainer.evaluator.generate_genotype(
            best_subnet, self.trainer.mutator)
        results = self.trainer.evaluator.query_result(
            genotype, cost_key='eval_acc1es')

        self.logger.info(f'Best subnet results: {results}')

        self.draw_random_process()
        return best_subnet

    def draw_random_process(self):
        splited_recoder = [[
            self.recorder[j][i] for j in range(len(self.recorder))
        ] for i in range(3)]
        import matplotlib.pyplot as plt
        x = list(range(len(self.recorder)))
        for i in range(3):
            plt.scatter(x, splited_recoder[i], marker='o', cmap='coolwarm')

        plt.savefig('./random_nb201_process.png')
