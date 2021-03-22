# compare standard bagged decision trees (bagging algorithm) with bagged decision tree with random undersampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier


def standard_bagging(X, y, cv):
    model = BaggingClassifier()
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('Standard Bagging', mean(scores)))


def undersample_bagging(X, y, cv):
    model = BalancedBaggingClassifier()
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('Random Undersampling Bagging', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_bagging(X, y, cv)
    undersample_bagging(X, y, cv)
