# compare xgboost with weighted xgboost
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from xgboost import XGBClassifier


# 20.08
def standard_xgboost(X, y, cv):
    model = XGBClassifier()
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<30} Mean ROC AUC: {1:.5f}'.format('Standard XGBoost', mean(scores)))


# 20.14
def weighted_xgboost(X, y, cv):
    model = XGBClassifier(scale_pos_weight=99)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<30} Mean ROC AUC: {1:.5f}'.format('Weighted XGBoost', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99],
                               flip_y=0, random_state=7)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_xgboost(X, y, cv)
    weighted_xgboost(X, y, cv)
