# compare standard neural network with weighted neural network
# NOTICE: need to install tensorflow and keras
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from keras.layers import Dense
from keras.models import Sequential

from plotDataset import plot_dataset
from collections import Counter
from numpy import array


def prepare_data():
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99],
                               flip_y=0, random_state=4)
    n_train = 5000
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


def define_model(n_input):
    model = Sequential()
    model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model


def standard_neural_network(n_input):
    model = define_model(n_input)
    model.fit(trainX, trainy, epochs=100, verbose=0)
    yhat = model.predict(testX)

    score = roc_auc_score(testy, yhat)
    print('{0:<30} ROC AUC: {1:.3f}'.format('Standard Neural Network', score))


def weighted_neural_network(n_input):
    model = define_model(n_input)
    weights = {0: 1, 1: 100}
    history = model.fit(trainX, trainy, class_weight=weights, epochs=100, verbose=0)
    yhat = model.predict(testX)

    score = roc_auc_score(testy, yhat)
    print('{0:<30} ROC AUC: {1:.3f}'.format('Weighted Neural Network', score))

    plot_dataset(testX, array([round(i[0]) for i in yhat]), Counter(array([round(i[0]) for i in yhat])))


if __name__ == '__main__':
    trainX, trainy, testX, testy = prepare_data()
    n_input = trainX.shape[1]

    plot_dataset(testX, testy, Counter(testy))

    standard_neural_network(n_input)
    weighted_neural_network(n_input)
