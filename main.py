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
model = DecisionTree()
model.fit(X_train.values, Y_train.values)
model.tree.show()



model.predict(X_test)
