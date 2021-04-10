from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # rysuje wykres wszystkich próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl,
                    edgecolor='black')


def my_plot(x, y):
    markers = ('s', 'x', 'o', '^', 'v', '8', '4')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'yellow', 'fuchsia')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    for idx, cl in enumerate(np.unique(y)):
        selection = []
        for value in y:
            if value == cl:
                selection.append(True)
            else:
                selection.append(False)
        plt.scatter(x=x[selection, 0], y=x[selection, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl,
                    edgecolor='black')
