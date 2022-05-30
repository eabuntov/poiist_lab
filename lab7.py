# Распознавание лиц с помощью нейронных сетей
# Необходимо установить следующие компоненты:
# pip install git+https://github.com/rcmalli/keras-vggface.git
# pip install mtcnn
# pip install keras-applications
# Заменить строку на from keras.utils.layer_utils import get_source_inputs
# В файле keras_vggface/models.py
from mtcnn import MTCNN
from matplotlib import pyplot
from PIL import Image
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine


def extract_faces(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)

    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    face_array = []
    for result in results:
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array.append(np.asarray(image))
    return face_array


# load the photo and extract the face
pixels = extract_faces('buntov.jpg')
# plot the extracted face
pyplot.imshow(pixels[1])
# show the plot
pyplot.show()

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filename):
    # extract faces
    faces = extract_faces(filename)
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


# define filenames
filename = 'buntov.jpg'
# get embeddings file filenames
embeddings = get_embeddings(filename)
# define sharon stone
buntov_id = get_embeddings('gt_db/buntov/Buntov.jpg')[0]
# verify known photos of sharon
print('Tests')
for embedding in embeddings:
    is_match(embedding, buntov_id)