# grid serch class weights with logistic regression for imbalance classfication
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=2)
    model = LogisticRegression(solver='lbfgs')
    # define grid
    balance = [{0: 100, 1: 1}, {0: 10, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 100}]
    param_grid = dict(class_weight=balance)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    # report the best configuration
    grid_result = grid.fit(X, y)
    print('Best: {0:f} using {1} \n'.format(grid_result.best_score_, grid_result.best_params_))
    # report all configuration
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('{0:f} ({1:f}) with: {2}'.format(mean, stdev, param))
