import numpy as np
import treelib

from .metrics.information_gain import information_gain


def find_best_feature(X: np.ndarray, y: np.ndarray, col_filter=[]):
    features_indices = np.size(X, 1)
    information_gains = [
        (i, information_gain(X, y, i)) for i in range(features_indices)
    ]
    return max(
        filter(lambda item: not item[0] in col_filter, information_gains),
        key=lambda item: item[1],
    )


def split(X: np.ndarray, y: np.ndarray, on: int) -> list:
    splitting_col = X[:, on]
    uniques = np.unique(splitting_col)
    split_data_indices = [(np.where(splitting_col == u)[0], u) for u in uniques]
    return [(X[indxs], y[indxs], value) for indxs, value in split_data_indices]


def count_unique_probs(arr: np.ndarray) -> dict:
    uniques, counts = np.unique(arr, return_counts=True)
    s = float(counts.sum())
    return {u: (c / s) for u, c in zip(uniques.tolist(), counts.tolist())}


class Classifier(object):
    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class DecisionTree(Classifier):
    def __init__(self):
        self.tree = treelib.Tree()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__fit(X, y, None, None, [])

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
                tag=f"Leaf",
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
                tag=f"Leaf",
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
            tag=f"Node {col_index}",
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
        #print (used_features)
        for split_X, split_y, parent_value in split_data:
            self.__fit(split_X, split_y, node, parent_value, used_features.copy())

    def rec_predict(self, X_test, node, index):
        if node.is_leaf():
            print ("prediction : " + str(node.data["prediction"]))
            return node.data["prediction"]
        tmp_node_attr = node.data["split_on_attr"]
        attr_value = X_test.at[index, 'Atr'+str(tmp_node_attr+1)]
        print(tmp_node_attr)
        print(attr_value)
        children = self.tree.children(node.identifier)
        child = next((child for child in children if child.data["parent_equals"] == attr_value), None)
        if not child:
            #"target_distribution": unique_prob_map,
            return None

        #print(child)
        return self.rec_predict(X_test, child, index)

    def predict(self, X_test):
        indices = list(X_test.index)
        #nodes = [node for node in self.tree.all_nodes() if not node.is_leaf()]
        #split_attr = [node.data["split_on_attr"] for node in self.tree.all_nodes() if not node.is_leaf()]
        predictions = [self.rec_predict(X_test, self.tree.get_node(self.tree.root), index) for index in indices]
        print (predictions)

    def transform(self, X):
        pass
