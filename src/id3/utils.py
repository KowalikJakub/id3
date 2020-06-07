__author__ = "Sebastian Puchała, Jakub Kowalik"

import pandas as pd
import numpy as np

from .metrics.information_gain import information_gain
from .metrics.information_gain import entropy


def find_best_feature(X: np.ndarray, y: np.ndarray, col_filter=[]):
    features_indices = np.size(X, 1)
    information_gains = [
        (i, information_gain(X, y, i)) for i in range(features_indices)
    ]
    return max(
        filter(lambda item: not item[0] in col_filter, information_gains),
        key=lambda item: item[1],
    )


def find_best_threshold(X: np.ndarray, y: np.ndarray, split_on: int):
    splitting_col = X[:, split_on]
    uniques, counts = np.unique(splitting_col, return_counts=True)
    uni = np.sort(uniques)
    inf_gains = []
    for t in uni:
        less_than_T, more_than_T = split_on_treshold(X, y, split_on, t)
        w1 = less_than_T[1].size / y.size
        w2 = more_than_T[1].size / y.size
        inf_gain = (
            entropy(y) - w1 * entropy(less_than_T[1]) - w2 * entropy(more_than_T[1])
        )
        inf_gains.append((inf_gain, t))

    best_threshold = max(inf_gains)
    return best_threshold[1]


def split(X: np.ndarray, y: np.ndarray, on: int) -> list:
    splitting_col = X[:, on]
    uniques = np.unique(splitting_col)
    split_data_indices = [(np.where(splitting_col == u)[0], u) for u in uniques]
    return [(X[indxs], y[indxs], value) for indxs, value in split_data_indices]


def split_on_treshold(X: np.ndarray, y: np.ndarray, split_on: int, T: int):
    splitting_col = X[:, split_on]
    uniques = np.unique(splitting_col)
    indxs = (np.where(splitting_col <= T))[0]
    indxs_2 = (np.where(splitting_col > T))[0]
    return (X[indxs], y[indxs], T, "<="), (X[indxs_2], y[indxs_2], T, ">")


def count_unique_probs(arr: np.ndarray) -> dict:
    uniques, counts = np.unique(arr, return_counts=True)
    s = float(counts.sum())
    return {u: (c / s) for u, c in zip(uniques.tolist(), counts.tolist())}


def test_accuracy(Y_test, predictions):
    Y_test = Y_test.values.reshape(Y_test.values.shape[0])
    predictions = predictions.values.reshape(Y_test.shape[0])
    accuracy = float(
        (np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T))
        / float(Y_test.size)
        * 100
    )
    return accuracy
