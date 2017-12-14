
import os;
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 3. Import libraries and modules
import numpy as np
#np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# other imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import glob
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from sklearn.metrics import classification_report
from skimage import io
from os import listdir
from os.path import isfile, join
import glob
import ntpath

def a2Test() :
    return 1;

def TrainModel(epochs, inputFile) :

    image_size = (224,224)
    # variables to hold features and labels
    features = []
    labels   = []
    Y = []
    class_count = 1000;
    allImages = []
    batchSize = 16
    totalCount = 0
    with open(inputFile) as myfile:
        totalCount = sum(1 for line in myfile)

    model = MobileNet(include_top=True,weights=None, classes = class_count);

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
 
    class_weight = [0.0] * class_count;
    class_weight[0] = 0.5
    class_weight[1] = 1

    model.fit_generator(generate_arrays_from_file(inputFile, batchSize, class_count),   epochs=epochs, verbose=1, steps_per_epoch=  int (( totalCount + batchSize - 1) / batchSize ), class_weight = class_weight )
    return model

