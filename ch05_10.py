from collections import Counter
from numpy import mean
from numpy import where
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot


def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Run it in terminal. In pycharm there will be an error
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    return scores


def plot_dataset(X, y):
    counter = Counter(y)
    print(counter)
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    plot_dataset(X, y)

    model = DummyClassifier(strategy='most_frequent')
    scores = evaluate_model(X, y, model)
    print('Mean Accuracy: %.2f%%' % (mean(scores) * 100))
