# compare SVM with SVM with calibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


def standard_svm(X, y, cv):
    model = SVC(gamma='scale')
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<40} Mean ROC AUC: {1:.3f}'.format('Standard SVM', mean(scores)))


# svm with calibrated probabilities
def cp_svm(X, y, cv):
    # model = SVC(gamma='scale')
    model = SVC(gamma='scale', class_weight='balanced')
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)

    # in fact, using calibrated instead of model to do cross validation has a worse result
    # scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    scores = cross_val_score(calibrated, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<40} Mean ROC AUC: {1:.3f}'.format('SVM with Calibrated Probabilities', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_svm(X, y, cv)
    cp_svm(X, y, cv)
