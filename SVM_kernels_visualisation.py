# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:21:22 2020

@author: Lucas
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