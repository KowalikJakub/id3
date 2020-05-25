import numpy as np

from id3.metrics.entropy import entropy

def information_gain(data: np.ndarray, reduced_data: np.ndarray): 
    return entropy(data) - entropy(reduced_data)