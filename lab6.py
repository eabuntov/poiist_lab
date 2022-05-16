from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from imutils import build_montages, paths
import numpy as np
import argparse
import imutils
import time
import cv2
import os

"""
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('test.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()
"""

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
    names = [p.split(os.path.sep)[-1] for p in imagePaths]
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
        name = imagePath.split(os.path.sep)[-1]
        # only process images that have a sufficient number of
        # examples belonging to the class
        if counts[names.index(name)] < minSamples:
            continue
        # perform face detection
        boxes = detect_faces(net, image, minConfidence)
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


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input directory of images")
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-components", type=int, default=563,
                help="# of principal components")
ap.add_argument("-v", "--visualize", type=int, default=-1,
                help="whether or not PCA components should be visualized")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the faces dataset
print("[INFO] loading dataset...")
(faces, labels) = load_face_dataset(args["input"], net,
                                    minConfidence=0.5, minSamples=20)


print("[INFO] {} images in dataset".format(len(faces)))
# flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])
# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# construct our training and testing split
split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
                         stratify=labels, random_state=42)
(origTrain, origTest, trainX, testX, trainY, testY) = split

# compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")
pca = PCA(
    svd_solver="randomized",
    n_components=args["num_components"],
    whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX)
end = time.time()
print("[INFO] computing eigenfaces took {:.4f} seconds".format(
    end - start))

# check to see if the PCA components should be visualized
if args["visualize"] > 0:
    # initialize the list of images in the montage
    images = []
    # loop over the first 16 individual components
    for (i, component) in enumerate(pca.components_[:16]):
        # reshape the component to a 2D matrix, then convert the data
        # type to an unsigned 8-bit integer so it can be displayed
        # with OpenCV
        component = component.reshape((62, 47))
        component = rescale_intensity(component, out_range=(0, 255))
        component = np.dstack([component.astype("uint8")] * 3)
        images.append(component)
    # construct the montage for the images
    montage = build_montages(images, (47, 62), (4, 4))[0]
    # show the mean and principal component visualizations
    # show the mean image
    mean = pca.mean_.reshape((62, 47))
    mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
    cv2.imshow("Mean", mean)
    cv2.imshow("Components", montage)
    cv2.waitKey(0)

# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(trainX, trainY)
# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print(classification_report(testY, predictions,
                            target_names=le.classes_))

# generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
# loop over a sample of the testing data
for i in idxs:
    # grab the predicted name and actual name
    predName = le.inverse_transform([predictions[i]])[0]
    actualName = le.classes_[testY[i]]
    # grab the face image and resize it such that we can easily see
    # it on our screen
    face = np.dstack([origTest[i]] * 3)
    face = imutils.resize(face, width=250)
    # draw the predicted name and actual name on the image
    cv2.putText(face, "pred: {}".format(predName), (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(face, "actual: {}".format(actualName), (5, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # display the predicted name  and actual name
    print("[INFO] prediction: {}, actual: {}".format(
        predName, actualName))
    # display the current face to our screen
    cv2.imshow("Face", face)
    cv2.waitKey(0)


# https://pyimagesearch.com/2021/05/10/opencv-eigenfaces-for-face-recognition/
