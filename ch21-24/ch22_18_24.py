# evaluate knn with uncalibrated probabilities for imbalanced classification
# grid search probability calibration with knn for imbalance classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV


def standard_knn(X, y, model, cv):
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print('{0:<30} Mean ROC AUC: {1:.3f}'.format('Standard KNN', mean(scores)))


def grid_search_knn(X, y, model, cv):
    calibrated = CalibratedClassifierCV(model)
    param_grid = dict(cv=[2, 3, 4], method=['sigmoid', 'isotonic'])
    grid = GridSearchCV(estimator=calibrated, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    grid_result = grid.fit(X, y)

    print('Grid Search Best: {0:f} using {1}'.format(grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('{0:f} ({1:f}) with: {2}'.format(mean, stdev, param))


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=4)
    model = KNeighborsClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    standard_knn(X, y, model, cv)
    grid_search_knn(X, y, model, cv)
