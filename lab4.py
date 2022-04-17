"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

import matplotlib
import matplotlib.pyplot
import sklearn

matplotlib.use('TKAgg')

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from skimage import io, segmentation as seg
from skimage.util import crop
from skimage.transform import resize
import numpy as np
from keras.datasets import mnist

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.
limit = 1000
# flatten the images
n_samples = len(train_X)
data = train_X.reshape((n_samples, -1))
label = test_X.reshape((len(test_X), -1))
# Create a classifier: a support vector classifier

knn = KNeighborsClassifier()

k_range = list(range(1, 9))
param_grid = dict(n_neighbors=k_range)

# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1, n_jobs=-1)

# fitting the model for grid search
grid_search = grid.fit(data[0:limit], train_y[0:limit])

print(grid_search.best_params_)

accuracy = grid_search.best_score_ * 100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

clfKNN = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
clfKNN.fit(data[0:limit], train_y[0:limit])
predicted = clfKNN.predict(label[0:limit])

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = matplotlib.pyplot.subplots(nrows=1, ncols=8, figsize=(12, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {clfKNN}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

matplotlib.pyplot.show()

pipelineSVC = make_pipeline(StandardScaler(), SVC(random_state=1))
param_grid_svc = [{
    'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
    'svc__kernel': ['linear']
},
    {
        'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__kernel': ['poly']
    },
    {
        'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__kernel': ['rbf']
    },
    {
        'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__kernel': ['sigmoid']
    }
]
gsSVC = GridSearchCV(estimator=pipelineSVC,
                     param_grid=param_grid_svc,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=-1)
# clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True), n_jobs=-1,
#                                            max_samples=1.0 / 8, n_estimators=8, verbose=True))


# Learn the digits on the train subset
gsSVC.fit(data[0:limit], train_y[0:limit])

print(gsSVC.best_score_)
print(gsSVC.best_params_)

clfSVC = gsSVC.best_estimator_
# Predict the value of the digit on the test subset
predicted = clfSVC.predict(label[0:limit])

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = matplotlib.pyplot.subplots(nrows=1, ncols=8, figsize=(12, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {clfSVC}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

matplotlib.pyplot.show()

image = io.imread('digits.jpg')
# plt.imshow(image)
labels = seg.slic(image, n_segments=11, compactness=10)
segments = np.ndarray((10, 28, 28))
i = 0
for section in np.unique(labels):
    rows, cols = np.where(labels == section)
    print("Image=" + str(section))
    print("Top-Left pixel = {},{}".format(min(rows), min(cols)))
    print("Bottom-Right pixel = {},{}".format(max(rows), max(cols)))
    segments[i] = resize(
        crop(image, ((min(rows), image.shape[0] - max(rows)), (min(cols), image.shape[1] - max(cols))), copy=True),
        (28, 28), anti_aliasing=True)
    i = i + 1
    print("---")
print(len(segments))
f, axarr = matplotlib.pyplot.subplots(2, 5)
for i in range(0, 5):
    for j in range(0, 2):
        index = j * 5 + i
        if index < len(segments):
            axarr[j, i].imshow(segments[j * 5 + i], cmap=matplotlib.pyplot.get_cmap('gray'))
        #axarr[j, i].axis('off')

matplotlib.pyplot.show()
data = sklearn.preprocessing.binarize(segments.reshape((10, -1)))
predicted = clfKNN.predict(data)

_, axes = matplotlib.pyplot.subplots(nrows=2, ncols=5, figsize=(12, 3))
for i, image, prediction in zip(range(0, 10), data, predicted):
    image = image.reshape(28, 28)
    axes[i % 2, i // 2].imshow(image, cmap=matplotlib.pyplot.get_cmap('gray'), interpolation="nearest")
    axes[i % 2, i // 2].set_title(f"Prediction: {prediction}")
    axes[i % 2, i // 2].set_axis_off()

matplotlib.pyplot.show()
