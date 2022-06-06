import cv2
import numpy as np
from keras.models import Model  # базовый класс для задания и обучения нейронной сети
from keras.layers import Input, Dense  # два типа слоя нейронной сети
from keras.utils import np_utils  # утилиты для быстрого кодирования значений истинности
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras.models import load_model
import os
from imutils import paths

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def detect_faces(net, image, minConfidence=0.5):
    # grab the dimensions of the image and then construct a blob
    # from it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    # pass the blob through the network to obtain the face detections,
    # then initialize a list to store the predicted bounding boxes
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > minConfidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # update our bounding box results list
            boxes.append((startX, startY, endX, endY))
    # return the face detection bounding boxes
    return boxes


def load_face_dataset(inputPath, net, minConfidence=0.5,
                      minSamples=15):
    # grab the paths to all images in our input directory, extract
    # the name of the person (i.e., class label) from the directory
    # structure, and count the number of example images we have per
    # face
    imagePaths = list(paths.list_images(inputPath))
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()
    # initialize lists to store our extracted faces and associated
    # labels
    faces = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]
        # only process images that have a sufficient number of
        # examples belonging to the class
        if counts[names.index(name)] < minSamples:
            continue
        # perform face detection
        try:
            boxes = detect_faces(net, image, minConfidence)
        except Exception:
            continue
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # extract the face ROI, resize it, and convert it to
            # grayscale
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (47, 62))
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            # update our faces and labels lists
            faces.append(faceROI)
            labels.append(name)
    # convert our faces and labels lists to NumPy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    # return a 2-tuple of the faces and labels
    return faces, labels


prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

(faces, labels) = load_face_dataset('gt_db', net,
                                    minConfidence=0.5, minSamples=14)

shape = faces.shape[0]
X_train = faces.reshape((shape, -1))

if os.path.isfile("FACES.h5"):
    model = load_model('FACES.h5')
else:
    batch_size = 128  # в каждой итерации рассматриваем сразу 128 обучающих примеров
    num_epochs = 50  # мы двадцать раз перебираем весь тренировочный набор
    hidden_size = 512  # в обоих скрытых слоях будет 512 нейронов

    num_train = 68  # в MNIST 60000 обучающих примеров
    num_test = 4  # в MNIST есть 10000 тестовых примеров

    height, width, depth = 47, 62, 1  # Изображения MNIST имеют размер 28x28 и оттенки серого.
    num_classes = 2  # имеется 10 классов (по 1 на цифру)

    X_train = X_train.astype('float32')
    X_train /= 255  # Нормализация данных в диапазоне [0, 1]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    Y_train = np_utils.to_categorical(labels, num_classes)  # Горячее кодирование меток

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
    model.save('FACES.h5')

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('buntov.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
width = np.mean(faces[:, 2])
faces = faces[faces[:, 2] >= width]
# Draw rectangle around the faces
faces2 = []
for (x, y, w, h) in faces:
    pict = np.asarray(gray[y:y + h, x:x + w])
    faces2.append(np.asarray(cv2.resize(pict, (47, 62))))
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

faces2 = np.array(faces2)
# Display the output

data = faces2.reshape((faces2.shape[0], -1))
predicted = model.predict(data)
x = np.argmax(predicted, axis=1)

le = LabelEncoder()
labels = le.fit_transform(labels)
i = 0
for (x, y, w, h) in faces:
    predName = le.inverse_transform(np.where(predicted[i] > 0.9))
    cv2.putText(img, "pred: {}".format(predName), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    i += 1
cv2.imshow('img', img)
cv2.waitKey()
