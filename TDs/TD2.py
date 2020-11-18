from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from sklearn.neural_network import MLPClassifier

X = [0, 1, 2, 3, 4, 5]

lab = []

for i in X:
    print(type(i))

for i in range(0, len(X)):
    value = X[i]
    if value % 2 == 0:
        lab.append(1)
    else:
        lab.append(0)

X = np.asarray(X)
lab = np.asarray(lab)

X = X.reshape(- 1, 1)

start_time = time.time()

svm_classifier = svm.SVC(kernel = 'rbf')
svm_fit = svm_classifier.fit(X, lab)
svm_pred = svm_classifier.predict(X)

MLP_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
MLP_classifier.fit(X, lab)
MLP_pred = MLP_classifier.predict(X)

print(time.time() - start_time)

print(confusion_matrix(lab, svm_pred))
print(confusion_matrix(lab, MLP_pred))

X = []

for i in range(0, 50):
    X.append(2 * i)
    X.append(2 * i + 1)

lab = []

for i in range(0, len(X)):
    value = X[i]
    if value % 2 == 0:
        lab.append(1)
    else:
        lab.append(0)

X = np.asarray(X)
X = X.reshape(- 1, 1)

start_time = time.time()

svm_fit = svm_classifier.fit(X, lab)
svm_pred = svm_classifier.predict(X)
MLP_classifier.fit(X, lab)
MLP_pred = MLP_classifier.predict(X)

print(time.time() - start_time)

print(confusion_matrix(lab, lab))
print(confusion_matrix(lab, MLP_pred))

X = []

for i in range(0, 5000):
    X.append(2 * i)
    X.append(2 * i + 1)

lab = []

for i in range(0, len(X)):
    value = X[i]
    if value % 2 == 0:
        lab.append(1)
    else:
        lab.append(0)

X = np.asarray(X)
X = X.reshape(- 1, 1)

start_time = time.time()

svm_fit = svm_classifier.fit(X, lab)
svm_pred = svm_classifier.predict(X)
MLP_classifier.fit(X, lab)
MLP_pred = MLP_classifier.predict(X)

print(time.time() - start_time)

print(confusion_matrix(lab, lab))
print(confusion_matrix(lab, MLP_pred))