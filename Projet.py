import numpy as np
import os
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
#
import pandas as pd
import pandas_tfrecords as pdtfr
from PIL import Image
import cv2
import base64
import io
from _datetime import datetime
import matplotlib.pyplot as plt

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

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 1000)

count = data.groupby('class').count()
print(count)


def labelReduction (data):
    count = data.groupby('class').count()
    count.reset_index(level=0, inplace=True)
    classes = []

    for i in range(0, 104):
        if count['id'][i]  > 200:
            var = count['class'][i]
            classes.append(var)

    print(classes)

    for i in range(0, len(data['class'])):
        if not data['class'][i] in classes:
            print(i, 'not in classes')
            data.drop([i], inplace = True)
        else:
            print(i, 'in classes')

def stringToRGB(base64_string):
    image = io.BytesIO(base64_string)
    image.seek(0)
    return Image.open(image)


def resize(image):
    basewidth = 24
    size = image.size[0]
    wpercent = (basewidth / int(size))
    hsize = int((float(image.size[1]) * float(wpercent)))
    img = image.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def imgShow(data):
    a = data['image'][50]
    a = stringToRGB(a)
    a.show()
    a = a.convert('L')
    a.show()
    a = np.asarray(a)
    a = a.flatten('F')
    print(a)
    print(a.shape)


def featureExtraction(data, datatest):
    global label, features, features_test, label_test
    label = []
    features = []
    features_test = []
    label_test = []

    for i in range(1, len(data)):
        classes = str(data['class'][i])
        label.append(classes)

    for i in range(1, len(data)):
        image = stringToRGB(data['image'][i])
        image = resize(image)
        image = np.asarray(image.convert('L'))
        image = image.flatten('F')
        features.append(image)

    for i in range(1, len(datatest)):
        classes_test = str(datatest['class'][i])
        label_test.append(classes_test)

    for i in range(1, len(datatest)):
        image_test = stringToRGB(datatest['image'][i])
        image_test = resize(image_test)
        image_test = np.asarray(image_test.convert('L'))
        image_test = image_test.flatten('F')
        features_test.append(image_test)

    features = np.array(features)
    features_test = np.array(features_test)
    label = np.array(label)
    label_test = np.array(label_test)


imgShow(datatest)

featureExtraction(data, datatest)

#label = label.astype(np.int)
#label_test = label_test.astype(np.int)

startTime = datetime.now()

spec = svm.SVC(kernel='linear')
fit = spec.fit(features, label)

print(datetime.now() - startTime)

pred = spec.predict(features_test)
print(metrics.classification_report(pred, label_test))


