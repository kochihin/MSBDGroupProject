
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
import glob
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

    
def CalculateValidationAccuracy(resultFile, validationFile) :
    tData = np.loadtxt(resultFile, dtype="str", delimiter=',' )
    vData = np.loadtxt(validationFile, dtype="str", delimiter='\t' )
    correct = 0
    total = 0
    result = {}
    for d in tData :
        result[d[0]] = d[1]

    for d in vData :
        if (int(result[d[0]]) == int(d[1])) :
            correct += 1
        total += 1

    print ("Accuracy :", correct * 100 / (total))

def CalculateValidationAccuracy2(resultFile, validationFile) :
    tData = np.loadtxt(resultFile, dtype="str", delimiter='\t' )
    vData = np.loadtxt(validationFile, dtype="str", delimiter=',' )
    correct = 0
    total = 0
    result = {}
    for d in tData :
        result[d[0]] = d[1]

    for d in vData :
        if (int(result[d[0]]) == int(d[1])) :
            correct += 1
        total += 1

    print ("Accuracy :", correct * 100 / (total))

def main() :


    tfInit();


    epochs = 5
    emRound = 30
    startRound = 1

    requirePreprocessing = True;
    requireEMTraining = True;    
    emInit = "./ImagePatch/emInit.txt"

    if requirePreprocessing :
        imagePreprocessing("./data", "./trimDark", "./ImagePatch", "./train.txt", emInit)
    
    if requireEMTraining :
        EMLoop(emInit, "./ImagePatch", "./ImagePatch/EMFinal.txt", emRound, epochs, startRound);


    epochs = 30


    print ("Training CNN - 2")
    upSample("./ImagePatch/EMFinal.txt", "./ImagePatch/EMFinal_us.txt", 4)
    model = TrainModel(epochs, "./ImagePatch/EMFinal_us.txt" );

    TestModel(model, "./ImagePatch", "./val.txt", "./Project3_validation_result.csv")
    CalculateValidationAccuracy("./Project3_validation_result.csv", "./val.txt")

    TestModel(model, "./ImagePatch", "./test.txt", "./Project3_result.csv")


main();
