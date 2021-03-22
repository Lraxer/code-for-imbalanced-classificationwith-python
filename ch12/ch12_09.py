# Oversample and plot imbalanced dataset with SMOTE
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where


def plot_dataset(X, y, counter):
    pyplot.figure()
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    counter = Counter(y)
    print(counter)
    plot_dataset(X, y, counter)

    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    counter = Counter(y)
    print(counter)
    plot_dataset(X, y, counter)
