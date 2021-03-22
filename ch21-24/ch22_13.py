# evaluate decision tree with uncalibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier


def standard_decision_tree(X, y, cv):
    model = DecisionTreeClassifier()
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<50} Mean ROC AUC: {1:.3f}'.format('Standard Decision Tree', mean(scores)))


def cp_decision_tree(X, y, cv):
    model = DecisionTreeClassifier()
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    scores = cross_val_score(calibrated, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<50} Mean ROC AUC: {1:.3f}'.format('Decision Tree with Calibrated Probabilities', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_decision_tree(X, y, cv)
    cp_decision_tree(X, y, cv)