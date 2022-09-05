from pplib.trainer import NB201Trainer


class RandomSearcher(object):

    def __init__(self, trainer: NB201Trainer = None, search_num: int = 1000):
        self.trainer = trainer
        self.search_num = search_num

    def search(self):
        best_val = 0
        best_subnet = None
        for _ in range(self.search_num):
            subnet = self.trainer.mutator.random_subnet
            self.trainer.mutator.set_subnet(subnet)
            val_loss = self.trainer.get_subnet_error(subnet)
            if val_loss > best_val:
                best_val = val_loss
                best_subnet = subnet

        return best_subnet
