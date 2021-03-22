from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where

sizes = [100, 1000, 10000, 100000]

for i in range(len(sizes)):
    n = sizes[i]
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    counter = Counter(y)
    print('Size=%d, Ratio=%s' % (n, counter))

    pyplot.subplot(2, 2, 1 + i)
    pyplot.title('n=%d' % n)
    pyplot.xticks([])
    pyplot.yticks([])

    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()

pyplot.show()
