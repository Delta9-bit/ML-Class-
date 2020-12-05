import numpy as np
import os
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pandas_tfrecords as pdtfr
from PIL import Image
import cv2
import base64
import io
from _datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

data = pdtfr.tfrecords_to_pandas("Data/Kaggle/train/00-192x192-798.tfrec.")
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/01-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/02-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/03-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/04-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/05-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/06-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/07-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/08-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/09-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/10-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/11-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/12-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/13-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/14-192x192-798.tfrec."), ignore_index=True)
data = data.append(pdtfr.tfrecords_to_pandas("Data/Kaggle/train/15-192x192-783.tfrec."), ignore_index=True)
datatest = pdtfr.tfrecords_to_pandas("Data/Kaggle/val/00-192x192-232.tfrec.")

np.set_printoptions(threshold=np.inf) # in-console display options
pd.set_option('display.max_rows', 1000) # in-console display options

count = data.groupby('class').count()
print(count)


def labelReduction (data): # keeps every class with 200+ observations
    count = data.groupby('class').count()
    count.reset_index(level=0, inplace=True)
    classes = []

    for i in range(0, 104):
        if count['id'][i]  > 200:
            var = count['class'][i]
            classes.append(var)

    for i in range(0, len(data['class'])):
        if not data['class'][i] in classes:
            data.drop([i], inplace = True)
        else:
            print('process running')


def stringToRGB(base64_string): # converts bytes to RGB/jpeg
    image = io.BytesIO(base64_string)
    image.seek(0)
    return Image.open(image)


def resize(image): # reduce resolution
    basewidth = 24
    size = image.size[0]
    wpercent = (basewidth / int(size))
    hsize = int((float(image.size[1]) * float(wpercent)))
    img = image.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def imgShow(data): #displays image
    a = data['image'][50]
    a = stringToRGB(a)
    a.show()


def featureExtraction(data, datatest): # convert jpeg into flatten array
    global label, features, features_test, label_test
    label = []
    features = []
    features_test = []
    label_test = []

    for i in range(0, len(data)):
        classes = str(data['class'][i])
        label.append(classes)

    for i in range(0, len(data)):
        image = stringToRGB(data['image'][i])
        image = resize(image)
        image = np.asarray(image)
        image = image.flatten('F')
        features.append(image)

    for i in range(0, len(datatest)):
        classes_test = str(datatest['class'][i])
        label_test.append(classes_test)

    for i in range(0, len(datatest)):
        image_test = stringToRGB(datatest['image'][i])
        image_test = resize(image_test)
        image_test = np.asarray(image_test)
        image_test = image_test.flatten('F')
        features_test.append(image_test)

    features = np.array(features)
    features_test = np.array(features_test)
    label = np.array(label)
    label_test = np.array(label_test)


# data pre-processing
imgShow(datatest)
labelReduction(data)
data.reset_index(inplace =True)
print(data)
featureExtraction(data, datatest)

features = features / 255.0
features_test = features_test / 255.0

runtime = []

# linear SVM
startTime = datetime.now()

lin_SVM = svm.SVC(kernel='linear', max_iter = 100)
lin_SVM_fit = lin_SVM.fit(features, label)

runtime.append(datetime.now() - startTime)

lin_pred = lin_SVM.predict(features_test)
print(metrics.classification_report(lin_pred, label_test))

# RBF SVM w/ gridsearch optimization
grid = {
	'C': [0.1, 1, 10, 100, 1000],
	'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
	'kernel': ['rbf']}

startTime = datetime.now()

rbf_SVM = svm.SVC(max_iter = 100)
grid_search = GridSearchCV(rbf_SVM, param_grid = grid, refit = True)
rbf_SVM_fit = grid_search.fit(features, label)

runtime.append(datetime.now() - startTime)

rbf_pred = grid_search.predict(features_test)
print(metrics.classification_report(rbf_pred, label_test))

# ANN
label = label.astype(int)

keras_MLP = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation = 'relu'),
    tf.keras.layers.Dense(104)]) # 1 hidden layer w/ 20 neurons

startTime = datetime.now()

# Opti controls
keras_MLP.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# Fit
keras_MLP.fit(features, label, epochs=10)

runtime.append(datetime.now() - startTime)
