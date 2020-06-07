__author__ = "Sebastian Puchała, Jakub Kowalik"


import pandas as pd
from src.id3 import *

data = pd.read_csv("data/divorce.csv", sep=";")

features = data.drop("Class", axis=1).astype("category")
target = data.Class


def test_id3(iter):
    acc = 0
    depth = 0
    size = 0
    for i in range(iter):
        X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
        model = DecisionTree_ID3()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        acc += test_accuracy(Y_test, predictions)
        depth += model.tree.depth()
        size += model.tree.size()
    parameters = {
        "acc": acc / iter,
        "depth": depth / iter,
        "size": size / iter,
    }
    return parameters


def test_id45_no_prune(iter):
    acc = 0
    depth = 0
    size = 0
    for i in range(iter):
        X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
        model = DecisionTree_C45(pruning=False)
        model.fit(X_train, Y_train)
        predictions = predict(X_test, model.tree)
        acc += test_accuracy(Y_test, predictions)
        depth += model.tree.depth()
        size += model.tree.size()
    parameters = {
        "acc": acc / iter,
        "depth": depth / iter,
        "size": size / iter,
    }
    return parameters


def test_id45(iter):
    acc = 0
    depth = 0
    size = 0
    for i in range(iter):
        X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
        model = DecisionTree_C45(pruning=True)
        model.fit(X_train, Y_train)
        predictions = predict(X_test, model.tree)
        acc += test_accuracy(Y_test, predictions)
        depth += model.tree.depth()
        size += model.tree.size()

    parameters = {
        "acc": acc / iter,
        "depth": depth / iter,
        "size": size / iter,
    }
    return parameters


def tree_comparision():
    X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
    model1 = DecisionTree_ID3()
    model1.fit(X_train, Y_train)
    model1.tree.show()
    model2 = DecisionTree_C45(pruning=False)
    model2.fit(X_train, Y_train)
    model2.tree.show()
    model3 = DecisionTree_C45(pruning=True)
    model3.fit(X_train, Y_train)
    model3.tree.show()


X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)

tree_comparision()

id45_no_prune_parameters = test_id45_no_prune(100)
id345_parameters = test_id45(100)

id3_parameters = test_id3(100)
print("parameters id3 :" + str(id3_parameters))
print("parameters id4.5 no prune :" + str(id45_no_prune_parameters))
print("parameters id4.5 :" + str(id345_parameters))
# predictions = predict(X_train, model.tree)  # model.predict(X_test)
# acc = test_accuracy(Y_train, predictions)

# print(str(acc) + "%")
# model.tree.show()
