import logging
import os

from torch.utils.tensorboard import SummaryWriter

from piconas.core.optims import Bananas, Npenas
from piconas.nas.search_spaces import get_search_space
from piconas.trainer import ZCTrainer
from piconas.utils import get_zc_benchmark_api, utils
from piconas.utils.pico_logging import get_logger

config = utils.get_config_from_args()

logger = get_logger('zc', log_file=f'{config.save}/log.log')
logger.setLevel(logging.INFO)

utils.log_args(config)

zc_params = sorted(config.search.zc_names)
out_dir = '-'.join(zc_params)

config.save = os.path.join(config.save, out_dir)

if not os.path.exists(config.save):
    os.makedirs(config.save)

writer = SummaryWriter(config.save)

# get_dataset_api(config.search_space, config.dataset) - unless it's for nb101 + mutation
dataset_api = None
zc_api = get_zc_benchmark_api(config.search_space, config.dataset)

search_space = get_search_space(config.search_space, config.dataset)
search_space.labeled_archs = [eval(arch) for arch in zc_api.keys()]
search_space.instantiate_model = False

supported_optimizers = {
    'bananas': Bananas(config, zc_api=zc_api),
    'npenas': Npenas(config, zc_api=zc_api),
}

utils.set_seed(config.seed)

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = ZCTrainer(optimizer, config, lightweight_output=True)
trainer.search(resume_from='', summary_writer=writer, report_incumbent=False)

logger.info('Ensemble experiment complete.')
logger.info('Done.')
