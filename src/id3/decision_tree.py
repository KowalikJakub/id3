import numpy as np
import treelib

from .metrics.information_gain import information_gain


def find_best_feature(X, y):
    features_indices = np.size(X, 1)
    information_gains = [
        (i, information_gain(X, y, i)) for i in range(features_indices)
    ]
    return max(information_gains, key=lambda item: item[1])


def split(X: np.ndarray, y: np.ndarray, on: int) -> list:
    splitting_col = X[:, on]
    uniques = np.unique(splitting_col)
    split_data_indices = [(np.where(splitting_col == u)[0], u) for u in uniques]
    _X = np.delete(X, on, 1)
    return [(_X[indxs], y[indxs], value) for indxs, value in split_data_indices]


class Classifier(object):
    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class DecisionTree(Classifier):
    def __init__(self):
        self.tree = treelib.Tree()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__fit(X, y, None, None)

    def __fit(self, _X, _y, caller_node: treelib.Node, parent_node_value):
        if np.all(_y == _y[0]):
            self.tree.create_node(
                tag=f"{caller_node.identifier}=={parent_node_value}",
                parent=caller_node,
            )
            return
        col_index, inf_gain = find_best_feature(_X, _y)
        caller_id = None if caller_node is None else caller_node.identifier
        node = self.tree.create_node(
            tag=f"{caller_id}=={parent_node_value}",
            data={
                "information_gain": inf_gain,
                "parent_node_value": parent_node_value,
            },
            parent=caller_node,
        )
        print(f"Node created {node.tag}")
        split_data = split(_X, _y, col_index)
        for split_X, split_y, parent_value in split_data:
            self.__fit(split_X, split_y, node, parent_value)

    def transform(self, X):
        pass
