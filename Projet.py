import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import pandas_tfrecords as pdtfr
from PIL import Image
import cv2
import base64
import io

data = pdtfr.tfrecords_to_pandas("Data/Kaggle/train/00-192x192-798.tfrec.")


def stringToRGB(base64_string):
    image = io.BytesIO(base64_string)
    image.seek(0)
    return Image.open(image)


a = data['image'][42]

a = stringToRGB(a)

# a.show()
a = np.asarray(a.convert('LA'))

label = data['class']

features = []

for i in range(1, len(data)):
    image = stringToRGB(data['image'][i])
    image = np.asarray(image.convert('LA'))
    features.append(image)

features = np.array(features)

print(features)
print(type(features))

