import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets, svm, metrics, model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#print(tf.__version__)

# keras MLP

# Importing data
MNIST = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = MNIST.load_data()

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print(train_images.shape)

# Normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

# Displays image
plt.imshow(train_images[1], cmap = plt.cm.binary)
plt.show()

# Model spec
keras_MLP = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(20, activation = 'relu'),
    tf.keras.layers.Dense(10)
])

# Opti controls
keras_MLP.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# Fit
keras_MLP.fit(train_images, train_labels, epochs=10)

# Computes accuracy
test_loss, test_acc = keras_MLP.evaluate(test_images, test_labels, verbose=2)

# Predictions on test set
predictions = keras_MLP.predict(test_images)

# Plot function
def plot_image(i, predictions, true, img):
    true, img = true[i], img[i]

    plt.imshow(img, cmap=plt.cm.binary)

    predicted = np.argmax(predictions)

    if predicted == true:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel(format(classes[predicted]), color = color)

# Actual plot
i = 0
plt.figure()
plot_image(i, predictions[i], test_labels, test_images)
plt.show()

# SVM

# Loading data
digits = datasets.load_digits()

images = digits.data
label = digits.target

# Test train split 80 - 20
X_train, X_test, y_train, y_test = model_selection.train_test_split(images, label, test_size = 0.2, random_state = 20)

# Model spec (linear)
spec = svm.SVC(kernel='linear')

# Fit
fit = spec.fit(X_train, y_train)

# Model spec (RBF)
grid = {'C': [1, 10, 100],
        'gamma': [0.1, 0.01, 0.001],
        'kernel': ['rbf']}

model = svm.SVC()
grid_spec = GridSearchCV(model, grid)

# Fit
grid_fit = grid_spec.fit(X_train, y_train)

# Prediction
pred_lin = spec.predict(X_test)
pred_RBF = grid_spec.predict(X_test)

# Accuracy

print(metrics.classification_report(pred_lin, y_test))
print(metrics.classification_report(pred_RBF, y_test))

# Plots

i = 0
plt.figure()
plot_image(i, pred_lin[i], y_test, X_test)
plt.show()