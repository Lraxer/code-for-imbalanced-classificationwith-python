# local outlier factor for imbalanced classificatoin
from numpy import vstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor


def lof_predict(model, trainX, testX):
    composite = vstack((trainX, testX))
    yhat = model.fit_predict(composite)
    return yhat[len(trainX):]


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999],
                               flip_y=0, random_state=4)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    model = LocalOutlierFactor(contamination=0.01)

    trainX = trainX[trainy == 0]
    yhat = lof_predict(model, trainX, testX)
    testy[testy == 1] = -1
    testy[testy == 0] = 1

    score = f1_score(testy, yhat, pos_label=-1)
    print('F1-measure: {:.3f}'.format(score))
