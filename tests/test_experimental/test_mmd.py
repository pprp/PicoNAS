import numpy as np
import torch
import torch.nn as nn


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the Gaussian kernel for domain adaptation.
    Reference: https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DAN/mmd.py
    """
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(
        total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the MMD-RBF (Maximum Mean Discrepancy with Radial Basis Function kernel) loss.
    Reference: https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DAN/mmd.py
    """
    batch_size = source.size(0)
    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the MMD-RBF loss without acceleration.
    Reference: https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DAN/mmd.py
    """
    batch_size = source.size(0)
    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss[0]


class LMMDLoss(nn.Module):
    """
    Deep Subdomain Adaptation Network for Image Classification.
    Reference: https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py
    """

    def __init__(
        self,
        class_num=31,
        kernel_type='rbf',
        kernel_mul=2.0,
        kernel_num=5,
        fix_sigma=None,
    ):
        super(LMMDLoss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size(0)
        weight_ss, weight_tt, weight_st = self.calculate_weight(
            s_label, t_label, class_num=self.class_num
        )
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += (
            torch.sum(weight_ss * SS)
            + torch.sum(weight_tt * TT)
            - torch.sum(2 * weight_st * ST)
        )
        return loss

    def calculate_weight(self, s_label, t_label, class_num=31):
        """s_label: shape is [4] t_label shape is [4, 3] where 3 is class_num"""
        s_batch_size = s_label.size(0)
        t_batch_size = t_label.size(0)

        s_sca_label = s_label.cpu().data.max(1)[1].numpy()
        s_vec_label = s_label.cpu().data.numpy()
        # breakpoint()
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        t_mask_arr = np.zeros((t_batch_size, class_num))
        t_mask_arr[:, index] = 1
        s_mask_arr = np.zeros((s_batch_size, class_num))
        s_mask_arr[:, index] = 1
        t_vec_label = t_vec_label * t_mask_arr
        s_vec_label = s_vec_label * s_mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])

        return (
            weight_ss.astype('float32'),
            weight_tt.astype('float32'),
            weight_st.astype('float32'),
        )


if __name__ == '__main__':
    lmmd_loss = LMMDLoss(class_num=3)
    source_label = torch.from_numpy(np.array([[0, 0, 2, 0]]))
    target_label = torch.from_numpy(
        np.array([[0.1, 0.8, 0.1], [0.4, 0.1, 0.5],
                 [0.1, 0.5, 0.4], [0.8, 0.1, 0.1]])
    )
    print(lmmd_loss.calculate_weight(source_label, target_label, class_num=3))
