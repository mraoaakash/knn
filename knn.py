import numpy as np
from builtins import object
from builtins import range
import cv2
import os 
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

    def predict(self, distance, k=1):
        n_test = distance.shape[0]
        predictions = np.zeros(n_test)
        for i in range(n_test):
            predictions[i] = np.argmax(np.bincount(self.y_train[np.argsort(distance[i, :])[:k]]))
        return predictions
    

train_path = "input/Training"
test_path = "input/Test"

X_train = []
y_train = []

for i in os.walk(train_path):
    for j in i[2]:
        img = cv2.imread(f"{i[0]}/{j}")
        X_train.append(img)
        y_train.append(i[0].split("/")[-1])
        pass

print(X_train)
print(y_train)