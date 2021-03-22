from sklearn.datasets import make_blobs
from numpy import *
import matplotlib.pyplot as plt


# create a dataset with a given class distribution
def get_dataset(proportions):
    # determine the number of classes
    n_classes = len(proportions)
    # determine the number of examples to generate for each class
    largest = max([v for k, v in proportions.items()])
    n_samples = largest * n_classes
    # create dataset
    # 这个函数本身总是产生元素数量相同的class
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1,
                      cluster_std=3)
    # collect the examples
    X_list, y_list = list(), list()
    for k, v in proportions.items():
        row_ix = where(y == k)[0]
        selected = row_ix[:v]
        X_list.append(X[selected, :])
        y_list.append(y[selected])
    return vstack(X_list), hstack(y_list)


def plot_dataset(X, y):
    n_classes = len(unique(y))
    for class_value in range(n_classes):
        row_ix = where(y == class_value)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    proportions = {0: 10000, 1: 100}
    X, y = get_dataset(proportions)
    plot_dataset(X, y)
