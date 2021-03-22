# undersample and plot imbalanced dataset with Tomek Links
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import TomekLinks
from plotDataset import plot_dataset

if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    counter = Counter(y)
    print(counter)
    plot_dataset(X, y, counter)

    undersample = TomekLinks(n_jobs=-1)
    X, y = undersample.fit_resample(X, y)

    counter = Counter(y)
    print(counter)
    plot_dataset(X, y, counter)
