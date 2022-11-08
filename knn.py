import numpy as np
from builtins import object as py_object
from builtins import range
import cv2
import os 
class knn(py_object):
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
            for j in range(n_test):
                distance[j, i] = np.sqrt(np.sum((X[j, :] - self.X_train[i, :])**2))
            pass
        return distance 

    def predict(self, distance, k=1):
        n_test = distance.shape[0]
        predictions = np.zeros(n_test)
        for i in range(n_test):
            closest = np.argsort(distance[i, :])[:k]
            predictions[i] = np.argmax(np.bincount(self.y_train[closest]))
        return predictions
    

train_path = ["./input/Training/Apple Red Delicious", "./input/Training/Avocado"]
test_path = ["./input/Test/Apple Red Delicious", "./input/Test/Avocado"]


def load_data(path):
    images = []
    labels = []
    for i in path:
        for j in os.listdir(i):
            img = cv2.imread(i + "/" + j)
            images.append(img)
            labels.append(path.index(i))
    return images, labels

x_train, y_train = load_data(train_path)
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = np.array(y_train)
y_train = y_train.reshape(y_train.shape[0])

x_test, y_test = load_data(test_path)
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0])

print(f"Training data loaded {np.array(x_train).shape}")
print(f"Training Labels loaded {np.array(y_train).shape}")
print(f"Testing data loaded {np.array(x_test).shape}")
print(f"Testing labels loaded {np.array(y_test).shape}")

model = knn()
model.train(x_train, y_train)
distance = model.distances(x_test)
predictions = model.predict(distance, k=1)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")