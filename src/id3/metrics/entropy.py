__author__ = "Sebastian Puchała, Jakub Kowalik"

import numpy as np


def entropy(data: np.ndarray):
    _, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    log_probs = np.log2(probs)
    return np.dot(-probs, log_probs)
