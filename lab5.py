import matplotlib.pyplot
import numpy
from keras.datasets import mnist  # подпрограммы для получения набора данных MNIST
from keras.models import Model  # базовый класс для задания и обучения нейронной сети
from keras.layers import Input, Dense  # два типа слоя нейронной сети
from keras.utils import np_utils  # утилиты для быстрого кодирования значений истинности
from lab5seg import segments
import tensorflow as tf
from keras.models import load_model
import os

"""gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3750)])
  except RuntimeError as e:
    print(e)"""

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
ordered_samples = numpy.asarray(segments('digits.jpg'))

matplotlib.pyplot.show()

shape = ordered_samples[0].shape[0]
data = ordered_samples[0].reshape((shape, -1))
data1 = ordered_samples[1].reshape((shape, -1))

if os.path.isfile("MNIST.h5"):
    model = load_model('MNIST.h5')
else:
    batch_size = 128  # в каждой итерации рассматриваем сразу 128 обучающих примеров
    num_epochs = 20  # мы двадцать раз перебираем весь тренировочный набор
    hidden_size = 512  # в обоих скрытых слоях будет 512 нейронов

    num_train = 60000  # в MNIST 60000 обучающих примеров
    num_test = 10000  # в MNIST есть 10000 тестовых примеров

    height, width, depth = 28, 28, 1  # Изображения MNIST имеют размер 28x28 и оттенки серого.
    num_classes = 10  # имеется 10 классов (по 1 на цифру)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # получить данные MNIST

    X_train = X_train.reshape(num_train, height * width)  # Свести данные к 1D
    X_test = X_test.reshape(num_test, height * width)  # Свести данные к 1D
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # Нормализация данных в диапазоне [0, 1]
    X_test /= 255  # Нормализация данных в диапазоне [0, 1]

    Y_train = np_utils.to_categorical(y_train, num_classes)  # Горячее кодирование меток
    Y_test = np_utils.to_categorical(y_test, num_classes)  # Горячее кодирование меток

    inp = Input(shape=(height * width,))  # Наши входные данные представляют собой одномерный вектор размером 784
    hidden_1 = Dense(hidden_size, activation='relu')(inp)  # Первый скрытый слой ReLU
    hidden_2 = Dense(hidden_size, activation='relu')(hidden_1)  # Второй скрытый слой ReLU
    out = Dense(num_classes, activation='softmax')(hidden_2)  # Выходной слой softmax

    model = Model(inputs=inp, outputs=out)  # Чтобы определить модель, просто укажем ее входной и выходной слои.

    model.compile(loss='categorical_crossentropy',  # использование функции потери кросс-энтропии
                  optimizer='adam',  # использование оптимизатора Adam
                  metrics=['accuracy'])  # сообщение о точности

    model.fit(X_train, Y_train,  # Обучить модель, используя тренировочный набор...
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1)  # удержание 10% данных для проверки
    model.evaluate(X_test, Y_test, verbose=1)  # Оценить обученную модель на тестовом наборе
    model.save('MNIST.h5')

x = [2, 5]
predicted = model.predict(data)
x[0] = numpy.argmax(predicted, axis=1)
predicted1 = model.predict(data1)
x[1] = numpy.argmax(predicted1, axis=1)
nrows = 2
_, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=5, figsize=(12, 3))
for j in range(2):
    for i, image, prediction in zip(range(5), ordered_samples[j], x[j]):
        image = image.reshape(28, 28)
        axes[j, i].imshow(image, cmap=matplotlib.pyplot.cm.gray_r, interpolation="nearest")
        axes[j, i].set_title(f"Segments: {prediction}")
        axes[j, i].set_axis_off()

matplotlib.pyplot.show()
