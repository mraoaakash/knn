import numpy as np
from builtins import object
from builtins import range


class knn(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def distances(self, X):
        n_train = self.X_train.shape[0]
        n_test = X.shape[0]
        distance = np.zeros((n_test, n_train))
        for i in range(n_train):
            distance[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))
            pass
        return distance 

    def predict_labels(self, distance, k=1):
        n_test = distance.shape[0]
        predictions = np.zeros(n_test)
        for i in range(n_test):
            predictions[i] = np.argmax(np.bincount(self.y_train[np.argsort(distance[i, :])[:k]]))
        return predictions
    