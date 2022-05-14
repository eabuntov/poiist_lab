"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

import matplotlib
import matplotlib.pyplot

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import sklearn.cluster
import numpy
from keras.datasets import mnist
import cv2

from lab5seg import segments

matplotlib.use('TKAgg')

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

limit = 10000
n_samples = len(train_X)
data = train_X.reshape((n_samples, -1))
label = test_X.reshape((len(test_X), -1))

knn = KNeighborsClassifier()

k_range = list(range(1, 10))
param_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, param_grid, cv=2, scoring='accuracy', return_train_score=False, n_jobs=-1)

grid_search = grid.fit(data[0:limit], train_y[0:limit])

print(grid_search.best_params_)

accuracy = grid_search.best_score_
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

clfKNN = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
clfKNN.fit(data[0:limit], train_y[0:limit])
predicted = clfKNN.predict(label[0:limit])



_, axes = matplotlib.pyplot.subplots(nrows=1, ncols=8, figsize=(12, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

matplotlib.pyplot.savefig('KNN_result.png', bbox_inches='tight')

print(
    f"Classification report for classifier {clfKNN}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)


disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix KNN")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
matplotlib.pyplot.savefig('KNN_matrix.png', bbox_inches='tight')


pipe = Pipeline([
    ('pca', PCA()),
    ('clf', KNeighborsClassifier()),
])

parameters = {
    'pca__n_components': [2, 3, 4, 5, 6, 7],
    }

gs = GridSearchCV(pipe, parameters, cv=2, n_jobs=-1, verbose=1)
gs.fit(data[0:limit], train_y[0:limit])

print("PCA Best score: %0.3f" % gs.best_score_)
print("Best parameters set:")
best_parameters = gs.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
clfPCA = gs.best_estimator_
predicted = clfPCA.predict(label[0:limit])

_, axes = matplotlib.pyplot.subplots(nrows=1, ncols=8, figsize=(12, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

matplotlib.pyplot.savefig('PCA_result.png', bbox_inches='tight')


print(
    f"Classification report for classifier {clfPCA}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)


disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix KNN+PCA")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
matplotlib.pyplot.savefig('PCA_matrix.png', bbox_inches='tight')



pipelineSVC = make_pipeline(StandardScaler(), SVC(random_state=1))
param_grid_svc = {
    'svc__C': [50.0, 75, 100, 125, 150],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

gsSVC = GridSearchCV(estimator=pipelineSVC,
                     param_grid=param_grid_svc,
                     scoring='accuracy',
                     cv=5,
                     refit=True,
                     n_jobs=-1)

gsSVC.fit(data[0:limit], train_y[0:limit])

print(gsSVC.best_score_)
print(gsSVC.best_params_)

clfSVC = gsSVC.best_estimator_

predicted = clfSVC.predict(label[0:limit])

_, axes = matplotlib.pyplot.subplots(nrows=1, ncols=8, figsize=(12, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

matplotlib.pyplot.savefig('SVC_result.png', bbox_inches='tight')

print(
    f"Classification report for classifier {clfSVC}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix SVC")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
matplotlib.pyplot.savefig('SVC_matrix.png', bbox_inches='tight')


ordered_samples = numpy.asarray(segments('digits.jpg'))

shape = ordered_samples[0].shape[0]
data = ordered_samples[0].reshape((shape, -1))
shape1 = ordered_samples[1].shape[0]
data = numpy.concatenate((data, ordered_samples[1].reshape((shape1, -1))), axis=0)

predicted = clfKNN.predict(data)
nrows = len(data)//5 + 1
_, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=5, figsize=(12, 3))
for i, image, prediction in zip(range(len(data)), data, predicted):
    image = image.reshape(28, 28)
    axes[i % nrows, i // nrows].imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    axes[i % nrows, i // nrows].set_title(f"Prediction KNN: {prediction}")
    axes[i % nrows, i // nrows].set_axis_off()

matplotlib.pyplot.savefig('Segments_classification_KNN.png', bbox_inches='tight')

predicted = clfPCA.predict(data)

_, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=5, figsize=(12, 3))
for i, image, prediction in zip(range(len(data)), data, predicted):
    image = image.reshape(28, 28)
    axes[i % nrows, i // nrows].imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    axes[i % nrows, i // nrows].set_title(f"Prediction PCA: {prediction}")
    axes[i % nrows, i // nrows].set_axis_off()

matplotlib.pyplot.savefig('Segments_classification_PCA.png', bbox_inches='tight')


predicted = clfSVC.predict(data)

print(predicted)

_, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=5, figsize=(12, 3))
for i, image, prediction in zip(range(0, 10), data, predicted):
    image = image.reshape(28, 28)
    axes[i % nrows, i // nrows].imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    axes[i % nrows, i // nrows].set_title(f"Prediction SVC: {prediction}")
    axes[i % nrows, i // nrows].set_axis_off()

matplotlib.pyplot.savefig('Segments_classification_SVC.png', bbox_inches='tight')
matplotlib.pyplot.show()


