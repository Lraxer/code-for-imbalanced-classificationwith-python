# roc curve for logistic regression model with optimal threshold
from numpy import sqrt
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot


def ch21_08():
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    trainX, testX, trainy, testy = train_test_split(X, y, train_size=0.5, random_state=2, stratify=y)
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)

    yhat = model.predict_proba(testX)
    yhat = yhat[:, 1]

    fpr, tpr, thresholds = roc_curve(testy, yhat)
    gmeans = sqrt(tpr * (1 - fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-mean=%.3f' % (thresholds[ix], gmeans[ix]))

    pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()

    pyplot.show()


def ch21_11():
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    trainX, testX, trainy, testy = train_test_split(X, y, train_size=0.5, random_state=2, stratify=y)
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)

    yhat = model.predict_proba(testX)
    yhat = yhat[:, 1]

    fpr, tpr, thresholds = roc_curve(testy, yhat)

    # Use Youden's J statistic
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold=%f' % (best_thresh))


if __name__ == '__main__':
    ch21_08()
    ch21_11()
