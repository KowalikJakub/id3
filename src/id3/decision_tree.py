import numpy as np
import treelib
import pandas as pd
import copy

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
    inf_gains= []
    for t in uni:
        less_than_T, more_than_T = split_on_treshold(X,y,split_on,t)
        w1 = less_than_T[1].size/y.size
        w2 = more_than_T[1].size/y.size
        inf_gain = entropy(y) - w1*entropy(less_than_T[1]) - w2*entropy(more_than_T[1])
        inf_gains.append((inf_gain,t))

    best_threshold = max (inf_gains)
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
    return (X[indxs], y[indxs], T, "<=") , (X[indxs_2], y[indxs_2], T, ">")

def count_unique_probs(arr: np.ndarray) -> dict:
    uniques, counts = np.unique(arr, return_counts=True)
    s = float(counts.sum())
    return {u: (c / s) for u, c in zip(uniques.tolist(), counts.tolist())}

def test_accuracy(Y_test, predictions):
    Y_test = Y_test.values.reshape(Y_test.values.shape[0])
    predictions = predictions.values.reshape(Y_test.shape[0])
    accuracy = float((np.dot(Y_test,predictions.T) +
            np.dot(1-Y_test,1-predictions.T))/float(Y_test.size)*100)
    return accuracy

def __predict(X_test, node, index, dec_tree):
    if node.is_leaf():
        return node.data["prediction"]
    tmp_node_attr = node.data["split_on_attr"]
    attr_value = X_test.at[index, 'Atr'+str(tmp_node_attr+1)]
    child = next((child for child in dec_tree.children(node.identifier)
            if (((attr_value <= child.data["threshold"] ) and (child.data["threshold_symbol"] == '<='))
            or  ((attr_value  > child.data["threshold"] ) and (child.data["threshold_symbol"] == '>')))
            ), None)
    if not child:
        if node.data["target_distribution"][0] > node.data["target_distribution"][1]:
            return 0
        else:
            return 1
    return __predict(X_test, child, index, dec_tree)

def predict(X_test, dec_tree):
    indices = list(X_test.index)
    predictions = [__predict(X_test, dec_tree.get_node(dec_tree.root), index, dec_tree)
                for index in indices]
    return pd.DataFrame(predictions, index = indices)

class Classifier(object):
    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


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
                tag=f"{parent_node_value} Leaf",
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
            tag=f"{parent_node_value} Node {col_index}",
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
        attr_value = X_test.at[index, 'Atr'+str(tmp_node_attr+1)]
        child = next((child for child in self.tree.children(node.identifier)
                if child.data["parent_equals"] == attr_value), None)
        if not child:
            if node.data["target_distribution"][0] > node.data["target_distribution"][1]:
                return 0
            else:
                return 1
        return self.__predict(X_test, child, index)

    def predict(self, X_test):
        indices = list(X_test.index)
        predictions = [self.__predict(X_test, self.tree.get_node(self.tree.root), index)
                    for index in indices]
        return pd.DataFrame(predictions, index = indices)

    def transform(self, X):
        pass


class DecisionTree_ID45(Classifier):
    def __init__(self, pruning = True):
        self.tree = treelib.Tree()
        self.pruning = pruning

    def fit(self, X , y) -> None:
        self.__fit(X.values, y.values, None, None,"", [])
        if self.pruning:
            self.prune(X,y)


    def __fit(
        self,
        _X,
        _y,
        caller_node: treelib.Node,
        parent_node_value: str,
        th_symbol:str,
        used_features: list,
    ):
        if np.all(_y == _y[0]):
            self.tree.create_node(
                tag=f"Leaf {th_symbol} {parent_node_value} ",
                data={
                    "parent_equals": parent_node_value,
                    "prediction": _y[0],
                    "used_features": used_features,
                    "threshold": parent_node_value,
                    "threshold_symbol": th_symbol,
                },
                parent=caller_node,
            )
            return

        if _X.shape[1] == len(used_features):
            unique_prob_map = count_unique_probs(_y)
            self.tree.create_node(
                tag=f"Leaf {th_symbol} {parent_node_value}",
                data={
                    "parent_equals": parent_node_value,
                    "target_distribution": unique_prob_map,
                    "used_features": used_features,
                    "threshold": parent_node_value,
                    "threshold_symbol": th_symbol,
                },
                parent=caller_node,
            )
            return

        col_index, inf_gain = find_best_feature(_X, _y, col_filter=used_features)
        th = find_best_threshold(_X, _y, col_index)
        used_features.append(col_index)
        unique_prob_map = count_unique_probs(_y)
        node = self.tree.create_node(
            tag=f"Node [{col_index}] {th_symbol} {parent_node_value} ",
            data={
                "information_gain": inf_gain,
                "parent_equals": parent_node_value,
                "target_distribution": unique_prob_map,
                "used_features": used_features,
                "split_on_attr": col_index,
                "threshold": parent_node_value,
                "threshold_symbol": th_symbol,
            },
            parent=caller_node,
        )
        split_data = split_on_treshold(_X, _y, col_index, th)
        for split_X, split_y, parent_value, th_symbol in split_data:
            self.__fit(split_X, split_y, node, parent_value,th_symbol, used_features.copy())



    def prune(self, X, y):
        prune_tree = copy.deepcopy(self.tree)
        leaves = [leaf for leaf in prune_tree.all_nodes() if leaf.is_leaf()]
        nodes = [node for node in prune_tree.all_nodes() if not node.is_leaf()]
        for leaf in leaves:
            prune_tree = copy.deepcopy(self.tree)
            try:
                parent = prune_tree.parent(leaf.identifier)
            except:
                continue
            if parent == None:
                continue
            tmp_tree = prune_tree.subtree(parent.identifier)
            predictions = predict(X, tmp_tree)
            acc = 1 - test_accuracy(y, predictions) /100.0
            e_0 = acc + np.sqrt(acc*(1-acc))/y.size
            predicton =  0 if parent.data["target_distribution"][0] > parent.data["target_distribution"][1] else 1
            th = parent.data["threshold"]
            th_sym = parent.data["threshold_symbol"]
            parent_data={
                "parent_equals": parent.data["parent_equals"],
                "prediction": predicton,
                "used_features": parent.data["used_features"],
                "threshold": th,
                "threshold_symbol":th_sym,
            }

            parent_tag= f"Leaf {th_sym} {th}"
            parent_parent= prune_tree.parent(parent.identifier)
            prune_tree.remove_node(parent.identifier)
            replacing_leaf = prune_tree.create_node(
                tag=parent_tag,
                data=parent_data,
                parent=parent_parent,
            )

            predictions = predict(X, prune_tree)
            acc2 = 1 - test_accuracy(y, predictions) /100.0
            e_1 = acc2 + np.sqrt(acc2*(1-acc2))/y.size
            if e_0 >= e_1:
                self.tree = prune_tree





        '''def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            super().fit(X, y)
            '''
