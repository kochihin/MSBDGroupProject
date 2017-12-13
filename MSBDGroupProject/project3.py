
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

def index_of_last_zero(lst):
    for i, value in enumerate(reversed(lst)):
        if value == 0:
            return len(lst)-i-1
    return -1

def index_of_last_nonzero(lst):
    for i, value in enumerate(reversed(lst)):
        if value != 0:
            return len(lst)-i-1
    return -1

def splitImage(srcDir, dstDir, imageSize) : 
#we need split the large image    

def EMTrain(epochs, dataFolder, inputFile, resultFile) :
#implement the EM training


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

def TestModel(model, preprocessFolder, testFile, resultFile) :
    
    maxPoolingThreshold = 0.5
    result_F = open(resultFile, 'w')
    input_F = open(testFile)
    for line in input_F:
        uid = line

        srcDir = preprocessFolder + "/" + uid ;
# find how many file to compare
        files = (glob.glob(srcDir + "/*.png"))

        IsLabelOne = False;

        for image_path in files :
            image_path = aLine[0];

            # create numpy arrays of input data
            # and labels, from each line in the file
            #x, y = process_line(line)
            #img = image.load_img(x)
            img = image.load_img(image_path)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            Y = model.predict(np.array(x))
            thePrediction = Y[0]
            LabelOnePercentage = float(thePrediction[1]) / float(thePrediction[0] + thePrediction[1])
            # maxPooling
            if (LabelOnePercentage > maxPoolingThreshold) :
                IsLabelOne = True;
                break;

            print (uid, LabelOnePercentage)

        if (IsLabelOne) :
            result_F.write(uid + ",1\n"  );
        else :
            result_F.write(uid + ",0\n"  );

    input_F.close();
    result_F.close();


def imagePreprocessing(dataFolder, trimDarkFolder, imagePatchFolder, trainFN, packFN):
    image_size = (224,224)

    #still need to implement: Cut the dark border for each image
    cutDark(dataFolder,trimDarkFolder);

    #still need to implement: Split the image into small image with size
    splitImage(trimDarkFolder, imagePatchFolder, image_size)

    #still need to implement: Assign label to small image
    initEMPercentage(imagePatchFolder, trainFN, packFN)

    #still need to implement: suffle the ratio
    suffleFile(packFN)

#should do EM for some loops    
def EMLoop(trainFN, imgFolder, finalFn, totalRound, epoch) :



    

def main() :

    tfInit();


main()
