import numpy as np
from builtins import object as py_object
from builtins import range
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.metrics import confusion_matrix
from math import sqrt


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

def cancer_dataset():
    # Link to the dataset used https://www.kaggle.com/code/lbronchal/breast-cancer-dataset-analysis/data
    dataset = pd.read_csv('dataset_input/data.csv')
    dataset = dataset[['diagnosis', 'area_mean', 'compactness_mean', 'symmetry_mean']]
    print("Dataset Before Renaming Cols:")
    print(dataset.head())

    dataset = dataset.replace({'diagnosis': {'M': 1, 'B': 0}})
    dataset.rename(columns = {'diagnosis':'y', 'area_mean':'x1', 'compactness_mean':'x2', 'symmetry_mean':'x2'}, inplace = True)
    print("Dataset After Renaming Cols:")
    print(dataset.head())
    x = np.array(dataset.iloc[:, 1:4].values)
    y = np.array(dataset.iloc[:, 0].values)
    shape = x.shape[0]
    x_train = x[:3*shape//4]
    y_train = y[:3*shape//4]
    x_test = x[3*shape//4:shape]
    y_test = y[3*shape//4:shape]

    print(f"Training data loaded {x_train.shape[0]}")
    print(f"Training Labels loaded {y_train.shape[0]}")
    print(f"Testing data loaded {x_test.shape[0]}")
    print(f"Testing labels loaded {y_test.shape[0]}")
    return x_train, y_train, x_test, y_test



def trainer(x_train, y_train, x_test, y_test, k):
    mode = knn()
    mode.train(x_train, y_train)
    distance = mode.distances(x_test)
    predictions = mode.predict(distance, k=k)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {round(accuracy, 4)} with k = {k}")
    return predictions, accuracy


def result_plotter(predictions, accuracy):
    sns.set_style("whitegrid")

    sns.barplot(x=["Malignant", "Benign"], y=np.bincount(y_train),orient="v", palette = sns.diverging_palette(220, 20,n=2)).set(title = "Training Data Distribution")
    plt.savefig("Training Data Distribution.png")
    plt.clf()

    sns.barplot(x=["Malignant", "Benign"], y=np.bincount(y_test),orient="v", palette=sns.diverging_palette(220, 20,n=2)).set(title = "Testing Data Distribution")
    plt.savefig("Testing Data Distribution.png")
    plt.clf()

    a = confusion_matrix(y_test, predictions)
    ax= plt.subplot()
    sns.heatmap(a, annot=True, fmt='g', ax=ax);
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Malignant', 'Benign']); ax.yaxis.set_ticklabels(['Benign', 'Malignant']);
    plt.savefig("Confusion Matrix.png")
    print("Plots Generated and Saved")


if __name__ == "__main__":
    accuracy_arr = []
    x_train, y_train, x_test, y_test = cancer_dataset()
    end = x_test.shape[0]+x_test.shape[0]
    end = int(sqrt(end))
    for k in range(1,end):
        predictions, accuracy = trainer(x_train, y_train, x_test, y_test,k)
        accuracy_arr.append(accuracy)
    sns.set_style("whitegrid")
    sns.lineplot(x=range(1,end), y=accuracy_arr, palette=sns.diverging_palette(220, 20,n=2))
    sns.scatterplot(x=range(1,end), y=accuracy_arr, palette=sns.diverging_palette(220, 20,n=2)).set(title = "Accuracy vs K")
    plt.savefig("Accuracy vs K.png")
    best_k = np.argmax(accuracy_arr)+1
    print(f"Best K value is {best_k}")
    predictions, accuracy = trainer(x_train, y_train, x_test, y_test, best_k)
    result_plotter(predictions, accuracy)
