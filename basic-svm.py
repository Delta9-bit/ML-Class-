#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:51:40 2020

@author: lucasA
"""

from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import random as rd
from matplotlib import pyplot as plt

rd.seed(1)

i = 0
cat =[]
x = []
y = []

for i in range(0, 150):
	f =  rd.randrange(0, 1000, 1) / 1000
	x.append(f)

	if f > 0.5:
		g = rd.gauss(800, 150) / 1000
		y.append(g)
		cat.append(1)
	else:
		g = rd.gauss(200, 100) / 1000
		y.append(g)
		cat.append(0)

	i += 1

var = list(zip(x, y))
var = pd.DataFrame(var)

cat = np.array(cat)

log_fit = LogisticRegression()
log_fit.fit(var, cat)

px, py = np.mgrid[-0.05:1:.01, -0.05:1:.01]
grid = np.c_[px.ravel(), py.ravel()]
probs = log_fit.predict_proba(grid)[:, 1].reshape(px.shape)

#svm_fit = svm.SVC(gamma = 0.001, C = 1000, kernel = 'linear')
svm_fit = svm.SVC(gamma = 0.001, C = 1000)
svm_fit.fit(var, cat)


df = pd.DataFrame(list(zip(x, y, cat)), columns = ['x', 'y', 'cat'])

groups = df.groupby('cat')

fig, ax = plt.subplots()

for cat, group in groups:
    ax.plot(group.x, group.y, marker = 'o', linestyle = '', ms = 5, label = cat)
ax.legend()

w = svm_fit.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 1)
yy = a * xx - (svm_fit.intercept_[0]) / w[1]

plt.ylim(0, 1.5)

plt.plot(xx, yy, 'k-', color = 'blue')



ax.contour(px, py, probs, levels = [.5], cmap = "Greys", vmin = 0, vmax = 0.6)

plt.show()






