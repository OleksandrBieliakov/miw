import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions
from plotka import my_plot


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
    
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)



def get_data():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1, stratify=y)
    return x_train, x_test, y_train, y_test


def train_reglog(train_data, train_data_classes_original, class_value):
    train_data_classes = train_data_classes_original.copy()
    train_data_classes[(train_data_classes_original == class_value)] = 1
    train_data_classes[(train_data_classes_original != class_value)] = 0
    reglog = LogisticRegressionGD()
    reglog.fit(train_data, train_data_classes)
    return reglog


def train_reglogs(train_data, train_data_classes):
    reglogs = []
    for class_value in np.unique(train_data_classes):
        reglog = train_reglog(train_data, train_data_classes, class_value)
        reglog.class_value = class_value
        reglogs.append(reglog)
    return reglogs


# returns true if perceptron was activated
def test_sample(sample, reglog):
    return reglog.predict(sample) == 1


def classify(reglogs, test_data):
    predictions = []
    for sample in test_data:
        activated_classes = []
        for reglog in reglogs:
            if test_sample(sample, reglog):
                activated_classes.append(reglog.class_value)
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
    reglogs = train_reglogs(train_data, train_data_classes)
    predictions = classify(reglogs, test_data)
    plot(test_data, test_data_classes)
    plot(test_data, predictions)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    #w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa)
    X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd.fit(X_train_01_subset, y_train_01_subset)
    plot_decision_regions(x=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    test()
