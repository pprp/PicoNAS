import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = [
    [80.86, 81.08, 80.91, 79.14, 81.69, 79.61],
    [82.23, 80.6, 80.23, 81.68, 82.44, 81.46],
    [82.69, 81.13, 82.53, 80.05, 79.12, 82.82],
]

b = [
    ['resnet20', 81.14],
    ['resnet32', 81.64],
    ['resnet56', 82.17],
]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
df = pd.DataFrame(np.array(a).T, columns=['resnet20', 'resnet32', 'resnet56'])
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
# 箱型图着色
# boxes → 箱线
# whiskers → 分位数与error bar横线之间竖线的颜色
# medians → 中位数线颜色
# caps → error bar横线颜色

df.plot.box(ylim=[78, 83], grid=True, color=color, ax=axes)
# color：样式填充
# df2 = pd.DataFrame(np.array(b), columns=['x', 'acc'])
# df2.plot(x='x',y='acc', ax=axes, kind='scatter', color='r')

plt.savefig('test.png')
