# decision tree evaluated on imbalanced dataset
# decision tree evaluated on imbalanced dataset with SMOTE oversampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# decision tree evaluated on imbalanced dataset
def withoutSMOTE(X, y):
    model = DecisionTreeClassifier()

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)  # n_jobs=-1
    print('Mean ROC AUC: %.3f' % mean(scores))


def useSMOTE(X, y):
    k_values = [1, 2, 3, 4, 5, 6, 7]
    for k in k_values:
        model = DecisionTreeClassifier()
        over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
        pipeline = Pipeline(steps=[('over', over), ('model', model)])

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        score = mean(scores)
        print('> k=%d, Mean ROC AUC: %.3f' % (k, score))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    withoutSMOTE(X, y)
    useSMOTE(X, y)
