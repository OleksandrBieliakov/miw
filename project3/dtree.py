import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from plotkab import plot_decision_regions
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # standardyzacja cech
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # entropy / gini
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=1)
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree4')
    plt.show()

    export_graphviz(tree, out_file='drzewo.dot', feature_names=['Długość płatka', 'Szerokość płatka'])

    forest = RandomForestClassifier(criterion='gini', n_estimators=15, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined,
    classifier=forest, test_idx=range(105,150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('randomforest')
    plt.show()


if __name__ == '__main__':
    main()
