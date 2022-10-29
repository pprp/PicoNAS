import random
import unittest
from unittest import TestCase

import model  # noqa: F401,F403
import torch
import torch.nn.functional as F
from model.mutable import MasterNet
from model.mutable.searchspace.search_space_xxbl import gen_search_space
# from model.mutable.basic_blocks import _remove_bn_layer_
from model.mutable.utils import PlainNet, create_netblock_list_from_str

from pplib.datasets import build_dataloader
# from model.mutable.utils import pretty_format
from pplib.predictor.pruners import predictive


class TestMutable(TestCase):

    def test_gen_search_space(self):
        # SuperResK1K5K1(in, out, stride, bottlenect_channel, sub_layers)
        plainnet_struct = 'SuperConvK3BNRELU(3,64,1,1)SuperResK1K5K1(64,168,1,16,3)SuperResK1K3K1(168,80,2,32,4)SuperResK1K5K1(80,112,2,16,3)SuperResK1K5K1(112,144,1,24,3)SuperResK1K3K1(144,32,2,40,1)SuperConvK1BNRELU(32,512,1,1)'
        net = MasterNet(plainnet_struct=plainnet_struct)
        student_blocks_list_list = gen_search_space(net.block_list, 0)
        print(student_blocks_list_list[0])

    def get_new_random_structure_str(self,
                                     the_net,
                                     get_search_space_func,
                                     num_replaces=1):

        # the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
        assert isinstance(the_net, PlainNet)
        selected_random_id_set = set()
        for replace_count in range(num_replaces):
            random_id = random.randint(0, len(the_net.block_list) - 1)
            if random_id in selected_random_id_set:
                continue
            selected_random_id_set.add(random_id)
            to_search_student_blocks_list_list = get_search_space_func(
                the_net.block_list, random_id)

            to_search_student_blocks_list = [
                x for sublist in to_search_student_blocks_list_list
                for x in sublist
            ]
            new_student_block_str = random.choice(
                to_search_student_blocks_list)

            if len(new_student_block_str) > 0:
                new_student_block = create_netblock_list_from_str(
                    new_student_block_str, no_create=True)
                assert len(new_student_block) == 1
                new_student_block = new_student_block[0]
                if random_id > 0:
                    last_block_out_channels = the_net.block_list[
                        random_id - 1].out_channels
                    new_student_block.set_in_channels(last_block_out_channels)
                the_net.block_list[random_id] = new_student_block
            else:
                # replace with empty block
                the_net.block_list[random_id] = None
        pass  # end for

        # adjust channels and remove empty layer
        tmp_new_block_list = [x for x in the_net.block_list if x is not None]
        last_channels = the_net.block_list[0].out_channels
        for block in tmp_new_block_list[1:]:
            block.set_in_channels(last_channels)
            last_channels = block.out_channels
        the_net.block_list = tmp_new_block_list

        new_random_structure_str = the_net.split(split_layer_threshold=6)
        return new_random_structure_str

    def test_use_masternet(self):
        plainnet_struct = 'SuperConvK3BNRELU(3,64,1,1)SuperResK1K5K1(64,168,1,16,3)SuperResK1K3K1(168,80,2,32,4)SuperResK1K5K1(80,112,2,16,3)SuperResK1K5K1(112,144,1,24,3)SuperResK1K3K1(144,32,2,40,1)SuperConvK1BNRELU(32,512,1,1)'

        dataload_info = ['random', 3, 100]

        # print(pretty_format(plainnet_struct))
        net = MasterNet(plainnet_struct=plainnet_struct)
        init_structure_str = str(net)

        # net.block_list = _remove_bn_layer_(net.block_list)

        dataloader = build_dataloader(
            'cifar100', type='train', data_dir='./data/cifar')

        score = predictive.find_measures(
            net,
            dataloader,
            dataload_info=dataload_info,
            measure_names=['zen'],
            loss_fn=F.cross_entropy,
            device=torch.device('cpu'))

        print(score)

        new_structures_str = self.get_new_random_structure_str(
            net,
            get_search_space_func=gen_search_space,
            num_replaces=2,
        )

        new_net = MasterNet(plainnet_struct=new_structures_str)

        score = predictive.find_measures(
            new_net,
            dataloader,
            dataload_info=dataload_info,
            measure_names=['zen'],
            loss_fn=F.cross_entropy,
            device=torch.device('cpu'))

        print(score)


if __name__ == '__main__':
    unittest.main()
