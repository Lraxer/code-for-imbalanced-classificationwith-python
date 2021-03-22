# random oversampling
# example of evaluating a decision tree with random oversampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)

steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# cannot add n_jobs=-1 in virtual environment
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv)
score = mean(scores)
print('F-measure: %.3f' % score)
