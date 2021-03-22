from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss


def brier_skill_score(y, yhat, brier_ref):
    bs = brier_score_loss(y, yhat)
    return 1.0 - (bs / brier_ref)


X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

probabilities = [0.01 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('Baseline: Brier Score=%.4f' % (avg_brier))
brier_ref = avg_brier
print('Reference: Brier Score=%.4f' % brier_ref)

probabilities = [0.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('P(class1=0): Brier Score=%.4f' % (avg_brier))
bss = brier_skill_score(testy, probabilities, brier_ref)
print('P(class1=0): BSS=%.4f' % (bss))

probabilities = [1.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('P(class1=1): Brier Score=%.4f' % (avg_brier))
bss = brier_skill_score(testy, probabilities, brier_ref)
print('P(class1=1): BSS=%.4f' % (bss))

avg_brier = brier_score_loss(testy, testy)
print('Perfect: Brier Score=%.4f' % (avg_brier))
bss = brier_skill_score(testy, testy, brier_ref)
print('Perfect: BSS=%.4f' % (bss))
