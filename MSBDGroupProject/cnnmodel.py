
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

def createModel() :
    return 1;



def cnn():
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, input_shape=(80, 80, 3), activation= 'relu' , border_mode= 'same' ))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(Flatten())
    model.add(Dense(512, activation= 'relu' , W_constraint=maxnorm(1)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation= 'softmax' ))
    model.compile(loss= 'categorical_crossentropy' , optimizer='adam', metrics=[ 'accuracy' ])
    return model;
