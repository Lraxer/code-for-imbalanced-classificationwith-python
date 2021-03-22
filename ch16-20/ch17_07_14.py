# compare weighted decision tree with decision tree
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier


# 17.07 Decision tree without any modification
def standrd_decision_tree(X, y, cv):
    model = DecisionTreeClassifier()
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<30} ROC AUC: {1:.3f}'.format('Decision Tree', mean(scores)))


# 17.14 Weighted decision tree in heuristic way
def weighted_decision_tree(X, y, cv):
    model = DecisionTreeClassifier(class_weight='balanced')
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<30} ROC AUC: {1:.3f}'.format('Weighted Decision Tree', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=3)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standrd_decision_tree(X, y, cv)
    weighted_decision_tree(X, y, cv)
