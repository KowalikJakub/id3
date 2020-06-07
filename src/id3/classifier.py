__author__ = "Sebastian Puchała, Jakub Kowalik"


class Classifier(object):
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
