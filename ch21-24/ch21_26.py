# search thresholds for imbalanced classification
from numpy import arange, argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)

    yhat = model.predict_proba(testX)
    probs = yhat[:, 1]
    thresholds = arange(0, 1, 0.001)

    scores = [f1_score(testy, to_labels(probs, t)) for t in thresholds]
    ix = argmax(scores)
    print('Threshold=%.3f, F-measure=%.5f' % (thresholds[ix], scores[ix]))
