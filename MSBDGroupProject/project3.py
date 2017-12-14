
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

from fileProcessing import *
from imageProcessing import *
from modelTraining import *
from cnnmodel import *

def tfInit() :
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    

def main() :


    tfInit();


#    epochs = 5
#    emRound = 30

    epochs = 1
    emRound = 1

    imagePreprocessing("./data", "./trimDark", "./ImagePatch", "./train.txt", "./ImagePatch/emInit.txt")
    EMLoop("./ImagePatch/emInit.txt", "./ImagePatch", "./ImagePatch/EMFinal.txt", 30, epochs);


#   epochs = 30
    upSample("./ImagePatch/EMFinal.txt", "./ImagePatch/EMFinal_us.txt", 3)
    model = TrainModel(epochs, "./ImagePatch/EMFinal_us.txt" );
    TestModel(model, "./ImagePatch", "./test.txt", "./Project3_result.csv")


main();
