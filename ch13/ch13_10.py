# undersample and plot imbalanced dataset with the Condensed Nearest Neighbor Rule
# fit_resample is slow
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import CondensedNearestNeighbour
from plotDataset import plot_dataset
import time

if __name__ == '__main__':
    a = time.time()
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    counter = Counter(y)
    print(counter)
    plot_dataset(X, y, counter)

    undersample = CondensedNearestNeighbour(n_neighbors=1, n_jobs=-1)
    X, y = undersample.fit_resample(X, y)

    counter = Counter(y)
    print(counter)
    plot_dataset(X, y, counter)

    b = time.time()
    print(b-a)
