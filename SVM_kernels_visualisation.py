# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:21:22 2020

@author: Lucas
"""
from sklearn import svm
import numpy as np
import random as rd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

rd.seed(1)

i = 0
cat = []
x = []
y = []

for i in range(0, 150):
	f = rd.randrange(0, 1000, 1) / 100
	x.append(f)

	if f > 5:
		g = rd.gauss(800, 250) / 100
		y.append(g)
		cat.append(1)
	elif f < 6:
		g = rd.gauss(200, 250) / 100
		y.append(g)
		cat.append(0)

	i += 1

var = list(zip(x, y))

var = np.array(var)

cat = np.array(cat)

param_grid = {
	'C': [0.1, 1, 10, 100, 1000],
	'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
	'kernel': ['rbf']}

lin_model = svm.SVC()
gaussian_model = svm.SVC()

grid = GridSearchCV(gaussian_model, param_grid, refit=True, verbose=3)

grid_fit = grid.fit(var, cat)

print(grid.best_params_)

gauss_model = svm.SVC(C=1000, gamma=0.01)
gauss_fit = gauss_model.fit(var, cat)

lin_fit = lin_model.fit(var, cat)

grid_predictions = gauss_fit.predict(var)
lin_predictions = lin_model.predict(var)

print(classification_report(cat, grid_predictions))
print(classification_report(cat, lin_predictions))

x_min = var.min() - 1
x_max = var.max() + 1
y_min = var.min() - 1
y_max = var.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(x_min, x_max, 0.01))

for i, clf in enumerate((lin_model, gauss_model)):
	plt.subplots()
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
	plt.scatter(var[:, 0], var[:, 1], c=cat, cmap=plt.cm.RdBu,
				s=50, alpha=1, edgecolor='k')

plt.show()

# Test Git
