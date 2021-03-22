# Undersample imbalanced dataset with NearMiss-1, NearMiss-2, NearMiss-3
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from plotDataset import plot_dataset


def nearmiss1(X, y):
    undersample = NearMiss(version=1, n_neighbors=3)
    X, y = undersample.fit_resample(X, y)
    counter = Counter(y)
    print("NearMiss-1", counter)

    plot_dataset(X, y, counter)


def nearmiss2(X, y):
    undersample = NearMiss(version=2, n_neighbors=3)
    X, y = undersample.fit_resample(X, y)
    counter = Counter(y)
    print("NearMiss-2", counter)

    plot_dataset(X, y, counter)


def nearmiss3(X, y):
    undersample = NearMiss(version=3, n_neighbors_ver3=3)
    X, y = undersample.fit_resample(X, y)
    counter = Counter(y)
    print("NearMiss-3", counter)

    plot_dataset(X, y, counter)


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    counter = Counter(y)
    print(counter)
    plot_dataset(X, y, counter)

    nearmiss1(X, y)
    nearmiss2(X, y)
    nearmiss3(X, y)
