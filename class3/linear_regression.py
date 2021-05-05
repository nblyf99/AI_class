"""
Linear Regression
Use boston house price dataset
"""
import random

import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = load_boston()

data = dataset['data']
target = dataset['target']
columns = dataset['feature_names']

dataframe = pd.DataFrame(data)
dataframe.columns = columns
dataframe['price'] = target
# print(dataframe.corr())

# sns.heatmap(dataframe.corr())
# plt.show()

# RM：小区平均的卧室个数
# LSTAT：低收入人群在周围的比例

rm = dataframe['RM']
lstat = dataframe['LSTAT']
target = dataframe['price']


def model(x, w, b):
    # vectorized model
    return np.dot(x, w.T)+b


def loss(yhat, y):
    # numpy broadcast numpy广播方法
    return np.mean((yhat - y) ** 2)


def partial_w(x, y, yhat):
    return np.array([2 * np.mean((yhat - y) * x[0]), 2 * np.mean((yhat - y) * x[1])])


def partial_b(x, y, yhat):
    return 2 * np.mean((yhat - y))


w = np.random.random_sample(size=(1, 2))
b = np.random.random()
learning_rate = 1e-5
losses = []

for i in range(200):
    batch_loss = []
    for batch in range(len(rm)):
        # batch training
        index = random.choice(range(len(rm)))
        rm_x, lstat_x = rm[index], lstat[index]
        x = np.array([rm_x, lstat_x])
        y = target[index]

        yhat = model(x, w, b)
        loss_v = loss(yhat, y)

        batch_loss.append(loss_v)

        w = w + -1 * partial_w(x, y, yhat) * learning_rate
        b = b + -1 * partial_b(x, y, yhat) * learning_rate

        if batch % 100 == 0:
            print('Epoch: {} Batch: {}, loss: {}'.format(i, batch, loss_v))
    losses.append(np.mean(batch_loss))

predicate = model(np.array([19, 7]), w, b)
print(predicate)
