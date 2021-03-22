# varied types of random forest for imbalanced classification
# 10 13 15 18
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


# ch23_10
def standard_random_forest(X, y, cv):
    model = RandomForestClassifier(n_estimators=10)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<50} Mean ROC AUC: {1:.3f}'.format('Standard Random Forest', mean(scores)))


# ch23_13
def class_balanced_random_forest(X, y, cv):
    model = RandomForestClassifier(n_estimators=10, class_weight='balanced')
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<50} Mean ROC AUC: {1:.3f}'.format('Class Balanced Random Forest', mean(scores)))


# ch23_15
def bootstrap_class_balanced_random_forest(X, y, cv):
    model = RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample')
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<50} Mean ROC AUC: {1:.3f}'.format('Bootstrap Class Balanced Random Forest', mean(scores)))


# ch23_18
def random_undersampling_random_forest(X, y, cv):
    model = BalancedRandomForestClassifier(n_estimators=10)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<50} Mean ROC AUC: {1:.3f}'.format('Random Forest with Random Undersampling', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_random_forest(X, y, cv)
    class_balanced_random_forest(X, y, cv)
    bootstrap_class_balanced_random_forest(X, y, cv)
    random_undersampling_random_forest(X, y, cv)
