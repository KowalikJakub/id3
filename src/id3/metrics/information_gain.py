import numpy as np

from .entropy import entropy


def information_gain(X: np.ndarray, y: np.ndarray, split_on: int):
    splitting_col = X[:, split_on]
    uniques, counts = np.unique(splitting_col, return_counts=True)
    wieghts = counts / counts.sum()
    split_entropy = sum(
        [w * entropy(y[splitting_col == u]) for u, w in zip(uniques, wieghts)]
    )
    return entropy(y) - split_entropy
