import os
from imutils import paths
from keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout
from keras.models import Model
from keras.applications.densenet import DenseNet201
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def load_dataset(inputPath: str, inputPath2: str, IMAGE_SIZE = (224, 224)) -> tuple:
    imagePaths = list(paths.list_images(inputPath))
    images = []
    labels = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]
        pic = cv2.resize(image, IMAGE_SIZE)
        images.append(pic)
        labels.append(name)
    imagePaths = list(paths.list_images(inputPath2))
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]
        pic = cv2.resize(image, IMAGE_SIZE)
        images.append(pic)
        labels.append(name)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


epochs = 50
batch_size = 32
covid_path = 'covid\\COVID'
noncovid_path = 'covid\\non-COVID'

m = DenseNet201(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

covid_images, covid_labels = load_dataset(covid_path, noncovid_path)

covid_x_train, covid_x_test, covid_y_train, covid_y_test = train_test_split(
    covid_images, covid_labels, test_size=0.2)

outputs = m.output
outputs = AveragePooling2D((2,2), strides=(2,2))(outputs)
outputs = Flatten(name="flatten")(outputs)
outputs = Dense(128, activation="relu")(outputs)
outputs = Dropout(0.3)(outputs)
outputs = Dense(64, activation="relu")(outputs)
outputs = Dense(2, activation="softmax")(outputs)

model = Model(inputs=m.input, outputs=outputs)

for layer in m.layers:
    layer.trainable = False

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(covid_x_train, covid_y_train, batch_size=batch_size, validation_data=(covid_x_test, covid_y_test),
                    validation_steps=len(covid_x_test)/batch_size, steps_per_epoch=len(covid_x_train)/batch_size,
                    epochs=epochs)

predicted = model.predict(covid_x_test)

disp = ConfusionMatrixDisplay.from_predictions(covid_y_test, predicted)
disp.figure_.suptitle("Confusion Matrix DenseNet201")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.savefig('DenseNet_matrix.png', bbox_inches='tight')
