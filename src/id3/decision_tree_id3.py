__author__ = "Sebastian Puchała, Jakub Kowalik"

import numpy as np
import pandas as pd
import treelib

from .utils import *
from .classifier import Classifier


class DecisionTree_ID3(Classifier):
    def __init__(self):
        self.tree = treelib.Tree()

    def fit(self, X, y) -> None:
        self.__fit(X.values, y.values, None, None, [])

    def __fit(
        self,
        _X,
        _y,
        caller_node: treelib.Node,
        parent_node_value: str,
        used_features: list,
    ):
        if np.all(_y == _y[0]):
            self.tree.create_node(
                tag=f"{parent_node_value} Leaf -> prediction {_y[0]}",
                data={
                    "parent_equals": parent_node_value,
                    "prediction": _y[0],
                    "used_features": used_features,
                },
                parent=caller_node,
            )
            return

        if _X.shape[1] == len(used_features):
            unique_prob_map = count_unique_probs(_y)
            self.tree.create_node(
                tag=f"{parent_node_value} Leaf",
                data={
                    "parent_equals": parent_node_value,
                    "target_distribution": unique_prob_map,
                    "used_features": used_features,
                },
                parent=caller_node,
            )
            return

        col_index, inf_gain = find_best_feature(_X, _y, col_filter=used_features)
        used_features.append(col_index)
        unique_prob_map = count_unique_probs(_y)
        node = self.tree.create_node(
            tag=f"{parent_node_value} Node [{col_index}]",
            data={
                "information_gain": inf_gain,
                "parent_equals": parent_node_value,
                "target_distribution": unique_prob_map,
                "used_features": used_features,
                "split_on_attr": col_index,
            },
            parent=caller_node,
        )
        split_data = split(_X, _y, col_index)
        for split_X, split_y, parent_value in split_data:
            self.__fit(split_X, split_y, node, parent_value, used_features.copy())

    def __predict(self, X_test, node, index):
        if node.is_leaf():
            return node.data["prediction"]
        tmp_node_attr = node.data["split_on_attr"]
        attr_value = X_test.at[index, "Atr" + str(tmp_node_attr + 1)]
        child = next(
            (
                child
                for child in self.tree.children(node.identifier)
                if child.data["parent_equals"] == attr_value
            ),
            None,
        )
        if not child:
            if (
                node.data["target_distribution"][0]
                > node.data["target_distribution"][1]
            ):
                return 0
            else:
                return 1
        return self.__predict(X_test, child, index)

    def predict(self, X_test):
        indices = list(X_test.index)
        predictions = [
            self.__predict(X_test, self.tree.get_node(self.tree.root), index)
            for index in indices
        ]
        return pd.DataFrame(predictions, index=indices)
