from sklearn import datasets, metrics, model_selection, svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

data = datasets.load_digits()

print(data.data.shape)

print(type(data.data))

#plt.gray()
#plt.matshow(data.images[0])
#plt.show()
images = data.data
target = data.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(images, target, test_size=0.2, random_state=20)

spec = svm.SVC(kernel='linear')
fit = spec.fit(X_train, y_train)

param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']}

model = svm.SVC()
grid_spec = GridSearchCV(model, param_grid)
fit = grid_spec.fit(X_train, y_train)

pred_lin = spec.predict(X_test)
pred_rbf = grid_spec.predict(X_test)

print(metrics.classification_report(pred_lin, y_test))
print(metrics.classification_report(pred_rbf, y_test))

print(model_selection.cross_val_score(grid_spec, images, target))