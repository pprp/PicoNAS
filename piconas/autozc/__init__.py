import copy

from .binary_ops import *  # noqa: F403
from .unary_ops import *  # noqa: F403

available_zc_candidates = []
_zc_candidates_impls = {}


def zc_candidates(name, bn=True, copy_net=True, force_clean=True, **impl_args):

    def make_impl(func):

        def zc_candidates_impl(net, device, *args, **kwargs):
            if copy_net:
                net = copy.copy(net)
                # net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net
            ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc

                import torch
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _zc_candidates_impls
        if name in _zc_candidates_impls:
            raise KeyError(f'Duplicated zc_candidates! {name}')
        available_zc_candidates.append(name)
        _zc_candidates_impls[name] = zc_candidates_impl
        return func

    return make_impl


def get_zc_candidates(name, net, device, *args, **kwargs):
    return _zc_candidates_impls[name](net, device, *args, **kwargs)


def get_zc_function(name):
    return _zc_candidates_impls[name]
