# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

available_measures = []
_measure_impls = {}


def measure(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def measure_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_prunable_copy(bn=bn).to(device)
                # set model.train()
            else:
                net = net_orig
            ret = func(net, *args, **kwargs, **impl_args)
            if copy_net and force_clean:
                import gc

                import torch

                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        global _measure_impls
        if name in _measure_impls:
            raise KeyError(f'Duplicated measure! {name}')
        available_measures.append(name)
        _measure_impls[name] = measure_impl
        return func

    return make_impl


def calc_measure(name, net, device, *args, **kwargs):
    return _measure_impls[name](net, device, *args, **kwargs)


def load_all():
    from . import auto_prox  # noqa: F401
    from . import epe_nas  # noqa: F401
    from . import eznas  # noqa: F401
    from . import fisher  # noqa: F401
    from . import gd_nas  # noqa: F401
    from . import grad_norm  # noqa: F401
    from . import grasp  # noqa: F401
    from . import jacov  # noqa: F401
    from . import l2_norm  # noqa: F401
    from . import nwot  # noqa: F401
    from . import plain  # noqa: F401
    from . import snip  # noqa: F401
    from . import swap  # noqa: F401
    from . import synflow  # noqa: F401
    from . import te_nas  # noqa: F401
    from . import zen  # noqa: F401
    from . import zico  # noqa: F401


load_all()
