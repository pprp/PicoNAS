import torch
import numpy as np
# import mmd 
import piconas.predictor.pinat.cdp.mmd as mmd

# 假定的参数
class_num = 3  # 假设有3个类
source_dim = 10  # 假设输入特征的维度为10
batch_size = 4  # 假设每批处理4个样本
kernel_type = 'rbf'  # 核函数类型

# 模拟数据
source_data = torch.randn(batch_size, source_dim).cuda()  # 模拟源域数据
target_data = torch.randn(batch_size, source_dim).cuda()  # 模拟目标域数据

# 模拟K_percentile和labels
K_percentile = [0.33, 0.66, 1.0]  # 假设的百分位数界限，这里简化为3等分
source_labels = torch.rand(batch_size).cuda()  # 源域标签，这里用随机值模拟连续标签
target_labels = torch.rand(batch_size).cuda()  # 目标域标签，同样用随机值模拟


# 转换标签为one-hot编码
def one_hot_classification(K_percentile, labels):
    def classification(label, K_percentile):
        for j, percentile in enumerate(K_percentile):
            if j == len(K_percentile) - 1:
                return j
            if (label < K_percentile[j + 1]) and (percentile <= label):
                return j
    batch_size = labels.size()[0]
    one_hot_label = np.zeros((batch_size, len(K_percentile)), dtype=int)
    for i, label in enumerate(labels):
        class_num = classification(label.item(), K_percentile)
        one_hot_label[i][class_num] = 1
    return one_hot_label

s_labels_one_hot = one_hot_classification(K_percentile, source_labels)
t_labels_one_hot = one_hot_classification(K_percentile, target_labels)

# 转换为torch tensor
s_labels_one_hot_tensor = torch.from_numpy(s_labels_one_hot).float()
t_labels_one_hot_tensor = torch.from_numpy(t_labels_one_hot).float()

# 初始化LMMD损失计算类
lmmd_loss_calculator = mmd.LMMD_loss(class_num=class_num, kernel_type=kernel_type)

# 计算损失
loss = lmmd_loss_calculator.get_loss(source_data, target_data, s_labels_one_hot_tensor, t_labels_one_hot_tensor)

print(f"LMMD Loss: {loss.item()}")
