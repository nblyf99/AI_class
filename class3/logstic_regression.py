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
price = dataframe['price']
greater_than_most = np.percentile(price, 66)
dataframe['expensive'] = dataframe['price'].apply(lambda p: int(p > greater_than_most))
target = dataframe['expensive']

print(dataframe[:20])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model(x, w, b):
    return sigmoid(np.dot(x, w.T) + b)


def loss(yhat, y):
    return -np.sum(y*np.log(yhat) + (1 - y)*np.log(1 - yhat))


def partial_w(x, y, yhat):
    return np.array([np.sum((yhat - y) * x[0]), np.sum((yhat - y) * x[1])])


def partial_b(x, y, yhat):
    return np.sum((yhat - y))


w = np.random.random_sample((1, 2))
b = np.random.random()
learning_rate = 1e-5


epoch = 200


losses = []

for i in range(epoch):
    batch_loss = []
    for batch in range(len(rm)):
        index = random.choice(range(len(rm)))

        x = np.array([rm[index], lstat[index]])
        y = target[index]

        yhat = model(x, w, b)

        loss_v = loss(yhat, y)

        w = w + -1 * partial_w(x, y, yhat) * learning_rate
        b = b + -1 * partial_b(x, y, yhat) * learning_rate

        if batch % 100 == 0:
            print('Epoch: {} Batch: {}, loss: {}'.format(i, batch, loss_v))
    losses.append(np.mean(batch_loss))


random_test_indices = np.random.choice(range(len(rm)), size=100)
decision_boundary = 0.5

for i in random_test_indices:
    x1, x2, y = rm[i], lstat[i], target[i]
    predicate = model(np.array([x1, x2]), w, b)
    predicate_label = int(predicate > decision_boundary)

    print('RM: {}, LSTAT: {}, EXPENSIVE: {}, Predicate: {}'.format(x1, x2, y, predicate_label))
