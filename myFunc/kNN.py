import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


def kNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"

    distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]


class KNNClassifier:

    def __init__(self, k):
        """初始化KNN分类器"""
        assert k >= 1, "K musst be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集训练kNN分类器"""
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):

        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        # 特征值数量相同
        assert X_predict.shape[1] == self ._X_train.shape[1], \
            "the features must be equal"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个带预测的数据x, 返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature of x must be equal to X_train"
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k
