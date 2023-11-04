import json

import matplotlib.pyplot as plt
import numpy as np
from nas_201_api import NASBench201API as API
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from piconas.utils.rank_consistency import kendalltau, pearson, spearman

# from emq.utils.rank_consistency import kendalltau, pearson, spearman

VISUALIZE = True

# Read file
with open('/data/lujunl/pprp/bench/zc_nasbench201_layerwise.json', 'rb') as f:
    input_dict = json.load(f)

nb201_api = API(
    '/data/lujunl/pprp/bench/NAS-Bench-201-v1_1-096897.pth', verbose=False)

ds_target = 'cifar10'  # cifar100, ImageNet16-120
input_dict = input_dict[ds_target]
zc_target = 'fisher_layerwise'  # grad_norm_layerwise, grasp_layerwise, l2_norm_layerwise, plain_layerwise, snip_layerwise, synflow_layerwise

# Convert the dictionary to input features and target labels
x_train = []
y_train = []
for key, value in input_dict.items():
    x_train.append(value[zc_target])
    # y_train.append(value['gt'])
    # query gt by key
    gt = nb201_api.get_more_info(
        int(key), dataset=ds_target, hp='200')['test-accuracy']
    y_train.append(gt)

# breakpoint()

# preprocess to find max length
max_len = 0
for i in range(len(x_train)):
    if len(x_train[i]) > max_len:
        max_len = len(x_train[i])
# padding the list to the max length
for i in range(len(x_train)):
    x_train[i] = x_train[i] + [0] * (max_len - len(x_train[i]))

# Ratio of whole dataset
ratio = 1
x_train = x_train[:int(len(x_train) * ratio)]
y_train = y_train[:int(len(y_train) * ratio)]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

# Create a Gradient Boosting Regressor and fit it to the training data
gbdt_model = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
gbdt_model.fit(x_train, y_train)

# Use the trained gbdt_model to predict the test data
y_pred = gbdt_model.predict(x_test)

# Calculate the mean squared error (MSE) loss between the predicted and actual values
mse_loss = mean_squared_error(y_test, y_pred)

# Calculate the Kendall's tau, Spearman's rho, and Pearson's r
kendalltau_score = kendalltau(y_test, y_pred)
spearman_score = spearman(y_test, y_pred)
pearson_score = pearson(y_test, y_pred)

print(f'MSE loss: {mse_loss:.4f}')
print(f'Kendall\'s tau: {kendalltau_score:.4f}')
print(f'Spearman\'s rho: {spearman_score:.4f}')
print(f'Pearson\'s r: {pearson_score:.4f}')

# Compute deviance in test dataset
test_score = np.zeros((500, 1), dtype=np.float64)
for i, y_pred in enumerate(gbdt_model.staged_predict(x_test)):
    test_score[i] = gbdt_model.loss_(y_test, y_pred)

plt.figure(figsize=(15, 15))  # Increased the figure size

feature_importance = gbdt_model.feature_importances_

norm = plt.Normalize(feature_importance.min(), feature_importance.max())
cmap = plt.get_cmap('coolwarm')

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 1
colors = cmap(norm(feature_importance[sorted_idx]))

# save to csv file with pos and feature_importance
import pandas as pd

df = pd.DataFrame({'pos': pos, 'feature_importance': feature_importance})
df.to_csv('gbdt.csv', index=False)

plt.bar(pos, feature_importance[sorted_idx], align='center', color=colors)
plt.yticks(pos, np.arange(sorted_idx.shape[0]) + 1, fontsize=12)
plt.xlabel('Layer Index', fontsize=16)
plt.ylabel('Relative Importance', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')

plt.tight_layout()
plt.savefig('gbdt_beautify.png')

# if VISUALIZE:
#     # Visualize each tree in the GBDT
#     plot_tree(gbdt_model.estimators_[0, 0],)
#     plt.show()

#     # compute deviance in test dataset
#     test_score = np.zeros((500, 1), dtype=np.float64)
#     # The test error at each iterations can be obtained via the staged_predict method
#     # which returns a generator that yields the predictions at each stage.
#     for i, y_pred in enumerate(gbdt_model.staged_predict(x_test)):
#         test_score[i] = gbdt_model.loss_(y_test, y_pred)

#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title('Deviance')
#     plt.plot(np.arange(500), gbdt_model.train_score_)
#     plt.plot(np.arange(500), test_score)
#     plt.legend(loc='upper right')
#     plt.xlabel('Boosting Iterations')
#     plt.ylabel('Deviance')

#     # feature_importance = 100 * (
#     #     gbdt_model.feature_importances_ / max(gbdt_model.feature_importances_))
#     feature_importance = gbdt_model.feature_importances_
#     index = np.argsort(feature_importance)
#     pos = np.arange(index.shape[0]) + 1
#     # save to csv file with pos and feature_importance
#     import pandas as pd
#     df = pd.DataFrame({'pos': pos, 'feature_importance': feature_importance})
#     df.to_csv('gbdt.csv', index=False)

#     plt.subplot(1, 2, 2)
#     plt.barh(pos, feature_importance[index], align='center')
#     plt.yticks(pos, np.arange(index.shape[0]) + 1)
#     plt.xlabel('Relative Importance')
#     plt.title('Variable Importance')
#     plt.savefig('gbdt.png')