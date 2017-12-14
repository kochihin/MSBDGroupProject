
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
from cnnmodel import *


def a2Test() :
    return 1;



def upSample(inputFile, outFn, usRatio) :
    result_F = open(outFn, 'w')
    input_F = open(inputFile)
    for line in input_F:
        aLine = line.split('\t')
        label = int(aLine[1]);
        origWeight = float(aLine[2])


        if (label == 1 ) :
            for k in range(usRatio):
                result_F.write(aLine[0] + "\t" + aLine[1] + "\t" + str( origWeight ) + "\n" )  # python will convert \n to os.linesep
        else :
            result_F.write(aLine[0] + "\t" + aLine[1] + "\t" + str( origWeight ) + "\n" )  # python will convert \n to os.linesep

    input_F.close()
    result_F.close()

    suffleFile(outFn)

def normalize(inputFile, outFn) : 
    users = {}
    minWeightUser = {}
    maxWeightUserFn = {}
    maxWeightUser = {}
    removeUser = {}
    minWeightUserFn = {}

    input_F = open(inputFile)
    for line in input_F:
        aLine = line.split('\t')
        uid = aLine[0].split('_')[3]
        correctWeight = float(aLine[2])
        if (not uid in users) :
            users[uid] = 0.0
            minWeightUser[uid] = 9999999
            maxWeightUser[uid] = 0
            
        users[uid] += 1 
        if (correctWeight > maxWeightUser[uid]) :
            maxWeightUser[uid] = correctWeight ;
            maxWeightUserFn[uid] = aLine[0]

            
        if (correctWeight < minWeightUser[uid]) :
            minWeightUser[uid] = correctWeight;
            minWeightUserFn[uid] = aLine[0]

    input_F.close()


    result_F = open(outFn, 'w')
    input_F = open(inputFile)
    for line in input_F:
        aLine = line.split('\t')
        uid = aLine[0].split('_')[3]
        label = int(aLine[1]);
        newWeight = float(aLine[2]) / users[uid]
        if (label == 0) :
            newWeight = 1
        origWeight = float(aLine[2])


        if (label == 1 and users[uid] > 1 and minWeightUserFn[uid] == aLine[0]) :
            print("Skip", uid, origWeight, newWeight) 
        else :
            result_F.write(aLine[0] + "\t" + aLine[1] + "\t" + str( origWeight ) + "\n" )  # python will convert \n to os.linesep

    input_F.close()
    result_F.close()


def normalizeFinal(inputFile, outFn) : 
    users = {}
    minWeightUser = {}
    maxWeightUserFn = {}
    maxWeightUser = {}
    removeUser = {}
    minWeightUserFn = {}

    input_F = open(inputFile)
    for line in input_F:
        aLine = line.split('\t')
        uid = aLine[0].split('_')[3]
        correctWeight = float(aLine[2])
        if (not uid in users) :
            users[uid] = 0.0
            minWeightUser[uid] = 9999999
            maxWeightUser[uid] = 0
            
        users[uid] += 1 
        if (correctWeight > maxWeightUser[uid]) :
            maxWeightUser[uid] = correctWeight ;
            maxWeightUserFn[uid] = aLine[0]

            
        if (correctWeight < minWeightUser[uid]) :
            minWeightUser[uid] = correctWeight;
            minWeightUserFn[uid] = aLine[0]

    input_F.close()


    result_F = open(outFn, 'w')
    input_F = open(inputFile)
    for line in input_F:
        aLine = line.split('\t')
        uid = aLine[0].split('_')[3]
        label = int(aLine[1]);
        newWeight = float(aLine[2]) / users[uid]
        if (label == 0) :
            newWeight = 1
        origWeight = float(aLine[2])


        if (label == 1 and users[uid] > 1 and maxWeightUserFn[uid] != aLine[0]) :
            print("Skip", uid, origWeight, newWeight) 
        else :
            result_F.write(aLine[0] + "\t" + aLine[1] + "\t" + str( origWeight ) + "\n" )  # python will convert \n to os.linesep

    input_F.close()
    result_F.close() 


def generate_arrays_from_file(path, batchSize, class_count):
    batchCount = 0

    inputs = []
    targets = []
    weights = []
    while 1:
        f = open(path)
        for line in f:

            aLine = line.split('\t')

            image_path = aLine[0];
            label = int(aLine[1]);
            weight = float(aLine[2]);

            if (label == 0) :
                weight = 1 
            else :
                weight = weight * 2


            # create numpy arrays of input data
            # and labels, from each line in the file
            #x, y = process_line(line)
            #img = image.load_img(x)
            img = image.load_img(image_path)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            batchCount += 1
            
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label] = 1;

            targets.append(ground_truth)
            inputs.append(x[0])

            weights.append(weight)

            if batchCount >= batchSize:
                batchCount = 0
                X = np.array(inputs)
                y = np.array(targets)
                w = np.array(weights)

                inputs = []
                targets = []
                weights = []
                yield X, y              #, w

        f.close()


def EMTrain(epochs, dataFolder, inputFile, resultFile) :


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
    
    
    
    result_F = open(resultFile, 'w')
    input_F = open(inputFile)
    for line in input_F:

        aLine = line.split('\t')

        image_path = aLine[0];
        label = int(aLine[1]);

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
        if (label == 1) :
            ratio = float(thePrediction[1]) / float(thePrediction[0] + thePrediction[1])
        else :
            ratio = float(thePrediction[0]) / float(thePrediction[0] + thePrediction[1])

        result_F.write(image_path + "\t" + str(label) + "\t" + str( ratio )+ "\t" + str( thePrediction[0] )+ "\t" + str( thePrediction[1] ) + "\n" )  # python will convert \n to os.linesep

    input_F.close();
    result_F.close();

    del model

    
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

    model = MobileNet(include_top=True,weights='imagenet', classes = class_count);

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
        uid = line.rstrip()

        srcDir = preprocessFolder + "/" + uid ;
# find how many file to compare
        files = (glob.glob(srcDir + "/*.png"))

        IsLabelOne = False;

        for image_path in files :


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

def EMLoop(trainFN, imgFolder, finalFn, totalRound, epoch) :
    nextFn = imgFolder + "/EMRound1.txt"
    EMTrain(epoch, imgFolder, trainFN , nextFn);

    removeLastN = 4
    for k in range(1, totalRound):
        inFn = imgFolder +"/EMRound" + str(k) + ".txt"
        normalizeFn = imgFolder +"/EMRound" + str(k) + "n.txt"
        nextFn = imgFolder +"/EMRound" + str(k+1) + ".txt"

        print ("Maximize Round - remove lowest likeihood record with label 1")
        for n in range(removeLastN - 1) :
            tFn = imgFolder +"/tmp" + str(n) + ".tmp"
            normalize(inFn ,tFn);
            inFn = tFn

        print ("Store normalize data file ", normalizeFn)
        normalize(inFn ,normalizeFn);

        print ("Expectation Round " + str(k) + " - train the model and recacluate the label")
        EMTrain(epoch, imgFolder, normalizeFn ,nextFn);

    normalize(nextFn ,finalFn);
