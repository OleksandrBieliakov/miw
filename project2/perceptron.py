import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
from plotka import my_plot


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=1000):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


def get_data():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1, stratify=y)
    return x_train, x_test, y_train, y_test


def train_perceptron(train_data, train_data_classes_original, class_value):
    train_data_classes = train_data_classes_original.copy()
    train_data_classes[(train_data_classes != class_value)] = -1
    train_data_classes[(train_data_classes == class_value)] = 1
    ppn = Perceptron()
    ppn.fit(train_data, train_data_classes)
    return ppn


def train_perceptrons(train_data, train_data_classes):
    perceptrons = []
    for class_value in np.unique(train_data_classes):
        perceptron = train_perceptron(train_data, train_data_classes, class_value)
        perceptron.class_value = class_value
        perceptrons.append(perceptron)
    return perceptrons


# returns true if perceptron was activated
def test_sample(sample, perceptron):
    return perceptron.predict(sample) == 1


def classify(perceptrons, test_data):
    predictions = []
    for sample in test_data:
        activated_classes = []
        for perceptron in perceptrons:
            if test_sample(sample, perceptron):
                activated_classes.append(perceptron.class_value)
        predictions.append(activated_classes)
    return predictions


def plot(data, classes):
    my_plot(x=data, y=classes)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


def test():
    train_data, test_data, train_data_classes, test_data_classes = get_data()
    perceptrons = train_perceptrons(train_data, train_data_classes)
    predictions = classify(perceptrons, test_data)
    plot(test_data, test_data_classes)
    plot(test_data, predictions)


def main():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    # w perceptronie wyjście jest albo 1 albo -1
    y_train_01_subset[(y_train_01_subset == 0)] = -1
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X_train_01_subset, y_train_01_subset)

    plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=ppn)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()
    

if __name__ == '__main__':
    test()
