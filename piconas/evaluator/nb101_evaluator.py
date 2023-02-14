from typing import List

import nasbench
import torch.nn.functional as F
from nasbench import api
from nasbench_pytorch.model import Network as NBNetwork

from piconas.evaluator.base import Evaluator
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.rank_consistency import kendalltau, pearson, spearman

# nasbench_path = '/data/home/scv6681/run/github/nasbench/nasbench_only108.tfrecord'
# nb = api.NASBench(nasbench_path)

# # net_hash = nb.hash_iterator()
# # you can get hashes using nasbench.hash_iterator()
# m = nb.get_metrics_from_hash('fe4395df38d94c4fafb973a8b04bbd65')
# ops = m[0]['module_operations']
# adjacency = m[0]['module_adjacency']

# net = NBNetwork((adjacency, ops))

# print(net)

s50_0 = [
    '9bf6f79e80397de4f50e1ece2a89a26a', '0aa07f2ad9b17b4382ffb0071126a0ab',
    '2d49b28ff1ec41076435c1358751a280', '3756ce086e95c91e631f73ae21f9b104',
    'ed0d6606073574ab143c34de4d8fd912', '838d107a40f4b32f8cd2fc1510709903',
    '2c7959a7c86a2c59ebeb7cc80c7c158d', '4a6b2f259c83cb4d32b543376d60d432',
    '271e604672eb93b263a0a91117bc11b3', '1ff97e7a5f5b02a199d20ad35b310783',
    '19cf5058e237bdd06e270f93e205407c', 'd123436366642a9aed1116e5b0e0157a',
    '5cdc1659ea10013a21350317f919013d', 'f03cd1b16f985e15f2144adf7c91c35e',
    '0ddbd5b26c6b0fbeabf9e43374100698', '2b853e658d2bf0e3d8e93317f33fb3e2',
    '98e59e3455f57e59c27cc8c13c17ad8d', 'ddff7508377c9ee41dc52bcf99a81cc3',
    'dc0e5dd172ec8bb3dd64e668ce89abae', '16738d5804321afa43f95707c188b4c4',
    '8d7bb7fbe9eec56e13d7cff743bb1040', 'd235c910559d678acc6380be07ef42b2',
    '382bd653b6d55c0b39a60bbc38b30393', 'b6a8ebb11b1a3ca962c35f12ba5703a6',
    '78a72a61383c92d059f365b4582ff014', 'fee27d6cd959f07cdebc88897bd42cd5',
    'd5915446e7b9ff59d43b4f6b45fe0b9f', '53817afbd7851a90a46a9fc89210cbab',
    'cbc5d6be4e38826871003964c4a6ad4f', '1f7202b8fc60e78d1f1e20dd3b7d8199',
    '81433dd263c685655b5c1b50d8a1423d', 'c779d2bbdd5a70b2408f58a822cae737',
    '8b64049d7698f01404978315a161bf1b', '486d8fd1daaab4c534f7c7f2d29e1d72',
    '244c3c3b6588aaf3691f2edb52bc67fe', '0dcc7db296fe1b2cf6ffc7cde02bd4c1',
    '70efea823d419d6279f84337e3a4e9ad', 'b5846e14eadcd2c2e516d409d91a89f5',
    '64e0ca9f514c1b771c898dc9bafbc2d1', '9b0fed82af35b426ee2a5ced58b38928',
    '8bcb995a60a0b893cca901a3223d670e', 'cda2d3a89dbd2c2f1829fa5014268f4c',
    'cb57576bec1c215211cf1b950f2abac5', '353f77bf17b4b33f44b4a617a9b914c1',
    'fe8994ab45273965f1a2fe24c0ac1fb6', 'c903eda1cc0b9b51c48f3326962735d7',
    '8d4f8153d81d4102f3d6be7593eb6721', '8ddb83d7733952c4d68e9d4a6928a3ca',
    '2869a98f664126a1501033ac533c2c28', 'b58301e4a55de45076d584ab2baedfe8'
]

s50_1 = [
    '702d7d17b123ef623e24ed963253a892', '5905ce6558653c106696f2ca67a169aa',
    '3477cfc6a18bb5f425e416b7fa8366ab', 'a8c619b026ab5eaf631f26a3f3e67245',
    '56215d93782a043f09d486e705556310', 'ee432963eb789789442307283c08f725',
    '0b545485705505724cbf59af02de61bc', 'f51827dd8940012a37382bdf42dac427',
    '9c22c916290fc29001786910c5829add', 'e308ec1748edfe66cc9e59118663bc19',
    '0c181e336bd57735b068c67dd525607c', '9a522cb58bb944bacfaba07e941957db',
    'b4474723b3b5b90189eb6c62dd4a8ba5', '1a2d6b6e351fa899139fbd0ed88b9784',
    '621d7cc921b2e903a6dcbf093b436254', 'c5be66d9056923e12cd092c193900a5c',
    'bc6345216cffdc01e0fd668d3b54b91b', '020540de5d9439a2438cfc7d3b387a63',
    '33f1a5871da28d18d9152be245106879', '84553b279cad72321ac825297d117dc8',
    'bbda5d80d3b2195339b2adf604052965', '95577a2fc9b298034759c0d42bf62f17',
    '7752d69a982148ea7b53c2f341bfed3f', '137c971e30af906f7916ffaa79246b3b',
    'f2372ec18742c53468dd37bce68c940b', 'acf563485a89f13f9620b0a061c13647',
    '2d9db01d716f737b86a6f022614e810f', '9437e559cf5de129d4525808d62b5525',
    'ae4145768b867aa15a2211db20247368', '363136e6b7903f688b667ac7b2916a17',
    'b125d6cd32f73cfc21e4dfbaf7fbc6ce', '36f121e5393ad20da04530157e10c021',
    '0da93253d9e1d89cff26887172c65990', '68d3a357f1285fe41a4bf0b5cc728c4d',
    '8ab3fb5099f38345eb63bf429587f85f', '964751a96f5dd585e1995d7afed909d4',
    '92b5244dee0565a9c6b4628372e6c246', 'fa50c938afe7f3f9a79028b873071390',
    '2f618a7cc7593e13db4b31aaf0e6d453', '9a7f3358de5ce1953a0ece6452ce2f27',
    '5c65c0a233aeb6ed75a3469167bb2d5d', '393075fb17d3dc42d93d70abc4c4ff67',
    '3005f0f18e3bdad5ffccd2fbbb91bc0e', '6d5b99911de5b4c9bffcdabc436f65e1',
    '51832d14df0d1d5b0ef56dceeb63cae5', 'cdd30ad541e60f23e514c8ffa1e0876a',
    '2bcf857abace68cbe513edbb4613d8db', '88abf8b238ac6ef594f14cb068351ab5',
    '0676110ce768e90c394e1df714e27e31', '8429113efbc66782fb2614e3e4dbb9ce'
]

s50_2 = [
    '7011a37d0417bd19df84f9feb5c55045', 'eab2034733c9921d20441e7617ef82f7',
    'b148c5e43b9b1be81b8245c2bc0c1cf3', '967261d430382974d46b278d9d9032e9',
    '8b5662f4d26fc138948ed8aec597a480', 'd9fc7d83d845a3c19b1eb3264edf82af',
    '422d0ca937e42019935bef6d7aac79dc', '86124d59578c270d1b194a802f034688',
    'e0b2c6bc6aaa8f3fd98a844908b13848', 'dddffeaa14cff252f5d1bc1733bd7228',
    '80c1eb443dcee2545dd9dcc9657ffdf2', '2cb2d474e1f5e09d28f5830635f52dc4',
    'bf19fe9a6fd8f5242a1b4fba8d2340be', '903edd93a1c7dd6b2af38e3dadf1bae7',
    '43c1dd8b5359ad70cafc4292d7b39000', '3405cc29ab3da6f0dc198d2c35255275',
    '248db06ac14774fb000d4c15cd772df6', '0cf3933806eef6fae7fde4337f85a734',
    '2f3ef01a534a4e9d6508d3ba4f882067', '72c8ed745dacfa4cccb412cd69156212',
    '36b28f9e4318dc8cd788c207204575bd', '5eb4238a33489be8adfeab866f39f904',
    '66c824348788e3bbab134325a25a73a1', 'cb057ed1f265fd617fe8668a63741898',
    'c78c325fc28ab96ca19c8db8f823661b', '5a7395b30b841be6f81236b87930407e',
    '57fbe881993cc64fcdc43cccd2a51868', '0cb17ff94ef2000905b9247ddcf1c8ee',
    'caea33eac459b709070192bb5ad8f838', '23f612bc0cd91f6d250f1906753b7b0a',
    '437bbd5451377d78c73a2e963dd6aad1', '462cf5b02084f16c3906357d931bcd47',
    'cbb3f3cf6de7ed377b6f10ee7244d860', 'beabc88c43f552c1c2b29b3d43ccf186',
    'c2b25de0a8c96fa5ba1642ad22c637d2', 'd16d4a875c10135f0987b34faeedc54c',
    '92b9473da7908b0a8549449aaeaf3494', 'b0a4a506ad07d8f2bed0fa2396bf6461',
    '627cab47208ef0806af92716c175047b', '905d855d059f12ef3558a0eb2c64469a',
    '83fe6eeda3153c30a606ba3362e69f3b', '747cb4c0eadf3f18c3a971a6ac125644',
    '7125fa2836b9c58fd10fa2f60b39b1cb', '57429463c409f9ede07160b7862f56fe',
    '30abc1c45d2814d51555a79394579858', 'ec24402f8f066ec4372c1b870de0f94e',
    '6ff09b9aa50010612e21c31e300a9cf1', '82b0b8ae4998f247d49cca416e0f40a7',
    'd0f67450c5372ef8c8e84eb4f36942ae', '3c3c6862f05fe0accb0af75b4daf0565'
]

RANDOM_SAMPLED_HASHES = s50_0


class NB101Evaluator(Evaluator):
    """Evaluate the NB101 Benchmark

    Args:
        trainer (NB101Trainer): _description_
        num_sample (int, optional): _description_. Defaults to None.
        search_space (str, optional): _description_. Defaults to 'nasbench201'.
        dataset (str, optional): _description_. Defaults to 'cifar10'.
        type (str, optional): _description_. Defaults to 'eval_acc1es'.
    """

    def __init__(self,
                 trainer,
                 dataset: str = 'cifar10',
                 NB101_type: str = 'final_test_accuracy',
                 **kwargs):
        super().__init__(trainer=trainer, dataset=dataset)
        self.trainer = trainer
        self.NB101_type = NB101_type
        self.search_space = 'nasbench201'
        self.dataset = dataset

        if dataset == 'cifar10':
            self.num_classes = 10

        self.nb101_api = api.NASBench('/data/home/scv6681/run/github/nasbench/nasbench_only108.tfrecord')

    def query_nb101_result(self, _hash):
        """query the indictor by hash."""
        m = self.nb101_api.get_metrics_from_hash(_hash)
        return m[1][108][0][self.NB101_type]

    def get_nb101_model(self, _hash):
        m = self.nb101_api.get_metrics_from_hash(_hash)
        ops = m[0]['module_operations']
        adjacency = m[0]['module_adjacency']
        return NBNetwork((adjacency, ops))

    def compute_rank_by_predictive(self,
                                   dataloader=None,
                                   measure_name: List = ['flops']) -> List:
        """compute rank consistency by zero cost metric."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []

        if dataloader is None:
            from piconas.datasets import build_dataloader
            dataloader = build_dataloader('cifar10', 'train')

        for _hash in RANDOM_SAMPLED_HASHES:
            # query the true indicator by hash.
            results = self.query_nb101_result(_hash)
            true_indicator_list.append(results)

            # get the model by hash.
            model = self.get_nb101_model(_hash)
            dataload_info = ['random', 3, self.num_classes]

            # get predict indicator by predictive.
            score = find_measures(
                model,
                dataloader,
                dataload_info=dataload_info,
                measure_names=measure_name,
                loss_fn=F.cross_entropy,
                device=self.trainer.device)
            generated_indicator_list.append(score)

        return self.calc_results(true_indicator_list, generated_indicator_list)

    def calc_results(self, true_indicator_list, generated_indicator_list):

        kt = kendalltau(true_indicator_list, generated_indicator_list)
        ps = pearson(true_indicator_list, generated_indicator_list)
        sp = spearman(true_indicator_list, generated_indicator_list)
        return [kt, ps, sp]
