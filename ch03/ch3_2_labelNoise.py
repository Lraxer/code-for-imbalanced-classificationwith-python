from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where

noise = [0, 0.01, 0.05, 0.07]

for i in range(len(noise)):
    n = noise[i]
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=n, random_state=1)
    counter = Counter(y)
    print("Noise=%d%%, Ratio=%s" % (int(n * 100), counter))

    pyplot.subplot(2, 2, 1 + i)
    pyplot.title("noise=%d%%" % int(n * 100))
    pyplot.xticks([])
    pyplot.yticks([])

    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
pyplot.show()
