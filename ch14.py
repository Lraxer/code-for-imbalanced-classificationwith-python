# evaluates a decision tree model
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN


# 14.07 without sampling
def basic_method(X, y, model, cv):
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # print('%-20s Mean ROC AUC: %.3f' % ('Basic Method', mean(scores)))
    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('Basic Method', mean(scores)))


# 14.14 random oversampling and undersamping
def random_over_under_sampling(X, y, model, cv):
    over = RandomOverSampler(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)

    pipeline = Pipeline(steps=[('o', over), ('u', under), ('m', model)])
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # print('%-20s Mean ROC AUC: %.3f' % ('Random Sampling', mean(scores)))
    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('Random Sampling', mean(scores)))


# 14.17 SMOTE and random undersampling
def SMOTE_random_sampling(X, y, model, cv):
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under), ('m', model)]

    pipeline = Pipeline(steps=steps)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('SMOTE, Random Undersampling', mean(scores)))


# 14.21 SMOTE with Tomek Links
def SMOTE_TomekLinks(X, y, model, cv):
    resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))

    pipeline = Pipeline(steps=[('r', resample), ('m', model)])
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('SMOTE, Tomek Links', mean(scores)))


# 14.25 SMOTE with ENN
def SMOTE_ENN(X, y, model, cv):
    resample = SMOTEENN()

    pipeline = Pipeline(steps=[('r', resample), ('m', model)])
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('SMOTE, ENN', mean(scores)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    model = DecisionTreeClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    basic_method(X, y, model, cv)
    random_over_under_sampling(X, y, model, cv)
    SMOTE_random_sampling(X, y, model, cv)
    SMOTE_TomekLinks(X, y, model, cv)
    SMOTE_ENN(X, y, model, cv)
