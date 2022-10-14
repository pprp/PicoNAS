# out_file = "./log_out.txt"

# f_o = open(out_file, "w")
# with open("./log.txt", 'r') as f:
#     contents = f.readlines()
#     for content in contents:
#         if 'export subnet' in content:
#             f_o.write(content.split('subnet')[1])

import matplotlib.pyplot as plt

in_file = './log_out.txt'
list_of_dict = []
with open(in_file, 'r') as f:
    contents = f.readlines()
    for content in contents:
        list_of_dict.append(eval(content))

x = []
y_skip = []
y_enlarge = []

for i, subnet_dict in enumerate(list_of_dict):
    if i % 2 == 0:
        continue
    x.append(i)
    cnt_skip = 0
    cnt_enlarge = 0

    for k, v in subnet_dict.items():
        if v == 'skip_connect':
            cnt_skip += 1
        else:  # if v in 'spatial_sep_pool':
            cnt_enlarge += 1

    y_skip.append(cnt_skip)
    y_enlarge.append(cnt_enlarge)

plt.figure(figsize=(12, 8), dpi=100)
plt.grid(True, linestyle='--', alpha=0.8)

font = {'family': 'Times New Roman', 'size': '24'}

plt.rc('font', **font)
plt.plot(
    x,
    y_skip,
    color='r',
    mfc='white',
    linewidth=2,
    marker='o',
    linestyle=':',
    label='No Enhancer')
plt.plot(
    x,
    y_enlarge,
    color='g',
    mfc='white',
    linewidth=2,
    marker='^',
    linestyle='-',
    label='Enlarge Enhancer')
plt.yticks(fontproperties='Times New Roman', size=20)
plt.xticks(fontproperties='Times New Roman', size=20)

plt.legend()

plt.xlabel(
    'Number of Epochs', fontdict={
        'family': 'Times New Roman',
        'size': 27
    })
plt.ylabel(
    'Number of Operations', fontdict={
        'family': 'Times New Roman',
        'size': 27
    })
plt.show()
