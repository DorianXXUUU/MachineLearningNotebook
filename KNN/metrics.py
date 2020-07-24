import numpy as np


def accuracy_score(y_true, y_predict):
    """calculate the accuracy between t_true and y_predict"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "The size of y_true must be equal to y_predict"

    return sum(y_true == y_predict) / len(y_true)