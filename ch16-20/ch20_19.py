# grid search positive class weights with xgboost for imbalance classification
# our result does not match book's. In our result, the default parameter, 1, is the best
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier

if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99],
                               flip_y=0, random_state=7)
    model = XGBClassifier()
    weights = [1, 10, 25, 50, 75, 99, 100]
    param_grid = dict(scale_pos_weight=weights)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    grid_result = grid.fit(X, y)

    print('Best: {0:f} using {1}'.format(grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('{0:f} ({1:f}) with: {2}'.format(mean, stdev, param))