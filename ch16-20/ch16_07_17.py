# compare standard logistic regression with weighted logistic regression model on an imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression


def standard_logistic_regression(X, y, cv):
    model = LogisticRegression(solver='lbfgs')
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<55} Mean ROC AUC: {1:.3f}'.format('Standard Logistic Regression', mean(scores)))


def weighted_logistic_regression(X, y, cv):
    # use the inverse of the class distribution
    weights = {0: 0.01, 1: 1.0}
    model = LogisticRegression(solver='lbfgs', class_weight=weights)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<55} Mean ROC AUC: {1:.3f}'.format('Weighted Logistic Regression', mean(scores)))


def heuristic_weighted_logistic_regression(X, y, cv):
    """
    weighted logistic regression for class imbalance with heuristic weights
    """
    model = LogisticRegression(solver='lbfgs', class_weight='balanced') # set class_weight to balanced
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<55} Mean ROC AUC: {1:.3f}'.format('Weighted Logistic Regression with heuristic weights', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=2)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_logistic_regression(X, y, cv)
    weighted_logistic_regression(X, y, cv)
    heuristic_weighted_logistic_regression(X, y, cv)
