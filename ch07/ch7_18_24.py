from collections import Counter
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from matplotlib import pyplot


def plot_pr_curve(test_y, model_probs):
    pyplot.figure()
    no_skill = len(test_y[test_y == 1]) / len(test_y)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    pyplot.plot(recall, precision, marker='.', label='Logistic')

    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()


def plot_roc_curve(test_y, naive_probs, model_probs):
    pyplot.figure()
    fpr, tpr, _ = roc_curve(test_y, naive_probs)
    pyplot.plot(fpr, tpr, linestyle='--', label='No Skill')

    fpr, tpr, _ = roc_curve(test_y, model_probs)
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')

    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()


def cal_roc_pr():
    X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
    trainX, testX, trainy, testy = train_test_split(X, y, train_size=0.5, random_state=2, stratify=y)

    # No Skill Model
    model = DummyClassifier(strategy='stratified')
    model.fit(trainX, trainy)
    yhat = model.predict_proba(testX)
    naive_probs = yhat[:, 1]

    # calculate roc auc
    roc_auc = roc_auc_score(testy, naive_probs)
    print('No Skill ROC AUC %.3f' % roc_auc)

    # calculate precision-recall auc
    precision, recall, _ = precision_recall_curve(testy, naive_probs)
    auc_score = auc(recall, precision)
    print('No Skill PR AUC: %.3f' % auc_score)

    # Logistic Regression Model
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    yhat = model.predict_proba(testX)
    model_probs = yhat[:, 1]

    # calculate roc auc
    roc_auc = roc_auc_score(testy, model_probs)
    print('Logistic ROC AUC %.3f' % roc_auc)

    # calculate precision-recall auc
    precision, recall, _ = precision_recall_curve(testy, model_probs)
    auc_score = auc(recall, precision)
    print('Logisitic PR AUC: %.3f' % auc_score)

    plot_roc_curve(testy, naive_probs, model_probs)
    plot_pr_curve(testy, model_probs)

    # summarize the distribution of predicted probabilities
    yhat = model.predict(testX)
    print(Counter(yhat))
    pyplot.figure()
    pyplot.hist(model_probs, bins=100)
    pyplot.show()


if __name__ == '__main__':
    cal_roc_pr()
