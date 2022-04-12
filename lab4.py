"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports

import matplotlib
import matplotlib.pyplot
matplotlib.use('TKAgg')

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

#for i in range(2):
#    matplotlib.pyplot.subplot(330 + 1 + i)
#    matplotlib.pyplot.imshow(train_X[i], cmap=matplotlib.pyplot.get_cmap('gray'))
#    matplotlib.pyplot.show()

#digits = datasets.load_digits()


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
limit = 10000
# flatten the images
n_samples = len(train_X)
data = train_X.reshape((n_samples, -1))
label = test_X.reshape((len(test_X), -1))
# Create a classifier: a support vector classifier
clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True), n_jobs=-1,
                                            max_samples=1.0 / 8, n_estimators=8, verbose=True))


# Learn the digits on the train subset
clf.fit(data[0:limit], train_y[0:limit])

# Predict the value of the digit on the test subset
predicted = clf.predict(label[0:limit])

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = matplotlib.pyplot.subplots(nrows=1, ncols=8, figsize=(10, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

matplotlib.pyplot.show()
