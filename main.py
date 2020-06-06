import pandas as pd
import matplotlib.pyplot as plt
from src.id3 import *

data = pd.read_csv('data/divorce.csv', sep=';')

features = data.drop('Class', axis=1).astype('category')
target = data.Class


def simple_validation(features, target, train_sample_size):
    msk = np.random.rand(len(features)) < train_sample_size
    return features[msk],features[~msk],target[msk],target[~msk]


X_train, X_test, Y_train, Y_test = simple_validation(features, target, 0.8)
model = DecisionTree_ID45()
model.fit(features.values, target.values)
predictions = model.predict(X_test)
acc = test_accuracy(Y_test, predictions)
print (acc)
model.tree.show()
