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
from skimage import io, segmentation as seg
from skimage.util import crop
from skimage.transform import resize
from skimage.filters import threshold_otsu
import sklearn.cluster
import numpy
from keras.datasets import mnist
import cv2

matplotlib.use('TKAgg')

#path = 'digits.jpg'
#img = cv2.imread(path)

#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
#thresh = threshold_otsu(blur)
#img_otsu = blur < thresh

#matplotlib.pyplot.imshow(255 * img_otsu, cmap=matplotlib.pyplot.cm.gray_r)
#matplotlib.pyplot.show()

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
limit = 100
# flatten the images
n_samples = len(train_X)
data = train_X.reshape((n_samples, -1))
label = test_X.reshape((len(test_X), -1))
# Create a classifier: a support vector classifier

knn = KNeighborsClassifier()

k_range = list(range(1, 10))
param_grid = dict(n_neighbors=k_range)

# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=2, scoring='accuracy', return_train_score=False, n_jobs=-1)

# fitting the model for grid search
grid_search = grid.fit(data[0:limit], train_y[0:limit])

print(grid_search.best_params_)

accuracy = grid_search.best_score_
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

matplotlib.pyplot.savefig('KNN_result.png', bbox_inches='tight')

print(
    f"Classification report for classifier {clfKNN}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix KNN")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
matplotlib.pyplot.savefig('KNN_matrix.png', bbox_inches='tight')
#matplotlib.pyplot.show()

################################################################################
# KNN + PCA pipeline + GridSearch

pipe = Pipeline([
    ('pca', PCA()),
    ('clf', KNeighborsClassifier()),
])

parameters = {
    'pca__n_components': [2, 3, 4, 5, 6, 7],
    #'clf__C': [1, 10, 100],
    }

gs = GridSearchCV(pipe, parameters, cv=2, n_jobs=-1, verbose=1)
gs.fit(data[0:limit], train_y[0:limit])

print("PCA Best score: %0.3f" % gs.best_score_)
print("Best parameters set:")
best_parameters = gs.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
clfPCA = gs.best_estimator_
# Predict the value of the digit on the test subset
predicted = clfPCA.predict(label[0:limit])
###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = matplotlib.pyplot.subplots(nrows=1, ncols=8, figsize=(12, 3))
for ax, image, prediction in zip(axes, test_X, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

matplotlib.pyplot.savefig('PCA_result.png', bbox_inches='tight')
###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {clfPCA}:\n"
    f"{metrics.classification_report(test_y[0:limit], predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(test_y[0:limit], predicted)
disp.figure_.suptitle("Confusion Matrix KNN+PCA")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
matplotlib.pyplot.savefig('PCA_matrix.png', bbox_inches='tight')
#matplotlib.pyplot.show()
###############################################################################
# SVC pipeline
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

matplotlib.pyplot.savefig('SVC_result.png', bbox_inches='tight')
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
disp.figure_.suptitle("Confusion Matrix SVC")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
matplotlib.pyplot.savefig('SVC_matrix.png', bbox_inches='tight')
#matplotlib.pyplot.show()

# plt.imshow(image)

### Сегментация изображения на строки с использованием кластеризации

# сценарий использовать командой python имя_программы.py файл_изображения
# имя файла с исходным изображением
inp = 'digits.jpg'
# загрузить изображение в оттенках серого
src_image = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)

### Подготовка изображения

# инвертировать (при необходимости)
src_image = numpy.max(src_image) - src_image
# сгладить (размер ядра подбирается по размеру символов и уровню помех)
image = cv2.GaussianBlur(src_image, (7, 7), 0)
# бинаризовать (с использованием метода Оцу)
level, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)

### Выделение фрагментов

# анализ компонентов связности
# альтернативный способ с использованием функции findContours
# в режиме RETR_EXTERNAL выделяются только внешние контуры объектов (без вложенных)
# параметр CHAIN_APPROX_NONE отключает сглаживание контуров
contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
boxes = numpy.array([cv2.boundingRect(c) for c in contours[0]])

### Устранение аномалий

d = boxes[:, 3]  # высоты фрагментов
m = numpy.median(d)  # средняя высота фрагментов
e = numpy.abs(d - m)  # отклонения высот от среднего
me = numpy.median(e)  # среднее отклонение

# критерий отбора - отклонение не более чем n средних
good = boxes[e <= 3 * me]

### Изображения фрагментов
fragments = [image[y:y + h, x:x + w] for x, y, w, h in good]


# функция формирования образцов для распознавания
def fit_size(image, rsize=(28, 28), offset=2):
    # заготовка результата
    result = numpy.zeros(rsize, dtype=image.dtype)
    # исходный размер изображения
    h, w = image.shape
    if w > h:
        # вписываем по ширине
        tw = rsize[0] - offset * 2
        th = int(tw * h / w)
        px = offset
        py = (rsize[1] - th) // 2
    else:
        # вписываем по высоте
        th = rsize[1] - offset * 2
        tw = int(th * w / h)
        px = (rsize[0] - tw) // 2
        py = offset

    # tw, th - итоговый размер с учётом пропорций и рамок
    # px, py - положение в финальном изображении

    # масштабируем
    rimage = cv2.resize(image, (tw, th), cv2.INTER_AREA)
    # помещаем в финальное изображение
    result[py:py + th, px:px + tw] = rimage
    return result


# образцы для распознавания
samples = [fit_size(s) for s in fragments]

f, axarr = matplotlib.pyplot.subplots(2, 5)
for i in range(0, 5):
    for j in range(0, 2):
        index = j * 5 + i
        if index < len(samples):
            axarr[j, i].imshow(samples[index], cmap=matplotlib.pyplot.get_cmap('gray'))

matplotlib.pyplot.savefig('Segmentation_result.png', bbox_inches='tight')

### Формирование строк


# вертикальные координаты центров фрагментов
frag_y = good[:, 1] + 0.5 * good[:, 3]

# кластеризация вертикальных координат методом DBSCAN
# расстояние внутри кластера - на основе высоты символа
clu = sklearn.cluster.DBSCAN(0.1 * m, min_samples=2)
# признаки должны быть вектором, даже если признак один
clu.fit(frag_y.reshape(-1, 1))

# группировка фрагментов в строки на основе кластеризации
# lines содержит списки номеров фрагментов в каждой строке
lines = {}
for i, label in enumerate(clu.labels_):
    if label in lines:
        lines[label].append(i)
    else:
        lines[label] = [i]

# вычисление положения строк как среднего положений фрагментов
# line_y содержит пары значений (метка_строки, значение)
line_y = [(label, numpy.median([frag_y[i] for i in line])) for label, line in lines.items()]
# сортировка строк по положению
line_y = sorted(line_y, key=lambda item: item[1])
ordered_lines = [lines[label] for label, _ in line_y]

# горизонтальные координаты центров фрагментов
frag_x = good[:, 0] + 0.5 * good[:, 2]
# сортируем фрагменты в каждой строке по горизонтали
ordered_lines = [sorted(line, key=lambda idx: frag_x[idx]) for line in ordered_lines]

# финальный массив с образцами для распознавания
a = []
for line in ordered_lines:
    for idx in line:
        a.append(samples[idx])
ordered_samples = numpy.asarray(a)

shape = ordered_samples.shape[0]
data = ordered_samples.reshape((shape, -1))
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


