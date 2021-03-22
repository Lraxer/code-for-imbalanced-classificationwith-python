# compare weighted SVM with SVM
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC


# 18.08 svm without modification
def standard_svm(X, y, cv):
    model = SVC(gamma='scale')
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('Standard SVM', mean(scores)))


# 10.14 weighted svm
def weighted_svm(X, y, cv):
    model = SVC(gamma='scale', class_weight='balanced')
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('Weighted SVM', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_svm(X, y, cv)
    weighted_svm(X, y, cv)
