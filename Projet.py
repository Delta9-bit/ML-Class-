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
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from mpl_toolkits import mplot3d

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


def labelReduction (data, datatest): # keeps every class with 200+ observations
    count = data.groupby('class').count()
    count.reset_index(level=0, inplace=True)
    classes = []

    for i in range(0, 104):
        if count['id'][i]  > 500:
            var = count['class'][i]
            classes.append(var)

    for i in range(0, len(data['class'])):
        if not data['class'][i] in classes:
            data.drop([i], inplace = True)

    for i in range(0, len(datatest['class'])):
        if not datatest['class'][i] in classes:
            datatest.drop([i], inplace = True)


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
labelReduction(data, datatest)
data.reset_index(inplace =True)
datatest.reset_index(inplace =True)
featureExtraction(data, datatest)

# normalization
features = features / 255.0
features_test = features_test / 255.0

# creating list to keep track of runtimes
runtime = []

# linear SVM
startTime = datetime.now()

lin_SVM = svm.SVC(kernel='linear')
lin_SVM_fit = lin_SVM.fit(features, label)

runtime.append(datetime.now() - startTime)

lin_pred = lin_SVM.predict(features_test)
print(metrics.classification_report(lin_pred, label_test))

# RBF SVM w/ gridsearch optimization
grid = {
	'C': [0.1, 1, 10, 100],
	'gamma': [1, 0.1, 0.01, 0.001],
	'kernel': ['rbf']}

startTime = datetime.now()

rbf_SVM = svm.SVC(max_iter = 1000)
grid_search = GridSearchCV(rbf_SVM, param_grid = grid, refit = True)
rbf_SVM_fit = grid_search.fit(features, label)

runtime.append(datetime.now() - startTime)

rbf_pred = grid_search.predict(features_test)
print(metrics.classification_report(rbf_pred, label_test))

# gamma / c / error graph
C = [0.1, 1, 10, 100]
gamma = [1, 0.1, 0.01, 0.001]

scores = [x[1] for x in grid_search.cv_results_]
scores = np.array(scores).reshape(len(C), len(gamma))

for ind, i in enumerate(C):
    plt.plot(gamma, scores[ind], label='C: ' + str(i))
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Mean score')
plt.show()

# ANN (1 hidden layer, 20 neurons)
label = label.astype(int)

keras_MLP = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation = 'relu'),
    tf.keras.layers.Dense(5)]) # 1 hidden layer w/ 20 neurons

startTime = datetime.now()

# Opti controls
keras_MLP.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# Fit
keras_MLP.fit(features, label, epochs=10)

runtime.append(datetime.now() - startTime)


def CNNprocessing(data, datatest):
    global features_CNN, features_test_CNN, label, label_test, input_shape

    features_CNN = np.zeros(shape = (2791, 24, 24, 3))
    features_test_CNN = np.zeros(shape = (56, 24, 24, 3))

    label = []
    label_test = []

    for i in range(0, len(data)):
        image = stringToRGB(data['image'][i])
        image = image.convert('RGB')
        image = resize(image)
        image = np.asarray(image)
        features_CNN[i] = image
        classes = str(data['class'][i])
        label.append(classes)

    for i in range(0, len(datatest)):
        image_test = stringToRGB(datatest['image'][i])
        image_test = image_test.convert('RGB')
        image_test = resize(image_test)
        image_test = np.asarray(image_test)
        features_test_CNN[i] = image_test
        classes = str(datatest['class'][i])
        label_test.append(classes)

    label = np.asarray(label)
    label_test = np.asarray(label_test)
    label = label.astype(int)
    label_test = label_test.astype(int)

    for i in range(0, len(label)):
        if label[i] == 49:
            label[i] = 1
        elif label[i] == 4:
            label[i] = 2
        elif label[i] == 103:
            label[i] = 3
        else:
            label[i] = 4

    for i in range(0, len(label_test)):
        if label_test[i] == 49:
            label_test[i] = 1
        elif label_test[i] == 4:
            label_test[i] = 2
        elif label_test[i] == 103:
            label_test[i] = 3
        else:
            label_test[i] = 4

    img_rows = 24
    img_cols = 24

    if K.image_data_format() == 'channels_first':
        features_CNN = features_CNN.reshape(features_CNN.shape[0], 3, img_rows, img_cols)
        features_test_CNN = features_test_CNN.reshape(features_test_CNN.shape[0], 3, img_rows, img_cols)
        print(features_CNN.shape)
        input_shape = (3, img_rows, img_cols)
    else:
        features_CNN = features_CNN.reshape(features_CNN.shape[0], img_rows, img_cols, 3)
        features_test_CNN = features_test_CNN.reshape(features_test_CNN.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)


# standard CNN
# preprocessing data
CNNprocessing(data, datatest)

# normalization
features_CNN = features_CNN / 255.0
features_test_CNN = features_test_CNN / 255.0

batch_size = 100 # 100 data points fitted at each epoch
epochs = 10 # total number of iterations

CNN = tf.keras.Sequential() # CNN (32 filters + 1 dense layer w/ 50 neurons)
CNN.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24, 24, 3))) # convolutional layer
CNN.add(Flatten())
CNN.add(Dense(50, activation='relu')) # standard layer
CNN.add(Dense(5, activation='softmax')) # classification layer

CNN.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])

#fit
startTime = datetime.now()

history = CNN.fit(features_CNN, label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(features_test_CNN, label_test))
score = CNN.evaluate(features_test_CNN, label_test, verbose=0)

runtime.append(datetime.now() - startTime)

# CNN multiple layers
mult_CNN = tf.keras.Sequential()
mult_CNN.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24, 24, 3))) # First convolutional layer
mult_CNN.add(Conv2D(64, (3, 3), activation='relu')) # Second convolutional layer
mult_CNN.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling / averaging values on 2*2 pixel grids
mult_CNN.add(Dropout(0.25)) # dropping 25% of neurons to prevent overfitting
mult_CNN.add(Flatten())
mult_CNN.add(Dense(50, activation='relu')) # Standard dense layer w/ 128 neurons
mult_CNN.add(Dropout(0.5)) # dropping 50% of neurons to prevent overfitting
mult_CNN.add(Dense(5, activation='softmax'))
mult_CNN.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

#fit
startTime = datetime.now()

history = mult_CNN.fit(features_CNN, label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(features_test_CNN, label_test))
score = mult_CNN.evaluate(features_test_CNN, label_test, verbose=0)

runtime.append(datetime.now() - startTime)

# Convolutional layers optimization

filter = [2, 4, 8]
accuracy = []

for k in filter:
    for i in filter:
        mult_CNN = tf.keras.Sequential()
        mult_CNN.add(Conv2D(k, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=(24, 24, 3)))  # First convolutional layer
        mult_CNN.add(Conv2D(i, (3, 3), activation='relu'))  # Second convolutional layer
        mult_CNN.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling / averaging values on 2*2 pixel grids
        mult_CNN.add(Dropout(0.25))  # dropping 25% of neurons to prevent overfitting
        mult_CNN.add(Flatten())
        mult_CNN.add(Dense(50, activation='relu'))  # Standard dense layer w/ 128 neurons
        mult_CNN.add(Dropout(0.5))  # dropping 50% of neurons to prevent overfitting
        mult_CNN.add(Dense(5, activation='softmax'))
        mult_CNN.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         optimizer='adam',
                         metrics=['accuracy'])

        # fit
        startTime = datetime.now()
        
        history = mult_CNN.fit(features_CNN, label,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=2,
                               validation_data=(features_test_CNN, label_test))
        score = mult_CNN.evaluate(features_test_CNN, label_test, verbose=0)

        runtime.append(datetime.now() - startTime)

        accuracy.append(score[1])

# Plot
accuracy = np.asarray(accuracy)

filter = np.meshgrid(filter)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(filter, filter, accuracy)