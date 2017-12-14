
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
import fileProcessing
import imageProcessing



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

def cutDark(srcDir, dstDir) :
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    files = (glob.glob(srcDir + "/*.png"))

    for f in files :
        imgData = io.imread(f)
        rawFileName = ntpath.basename(f)

        iRow = np.max(imgData, axis = 1)
        iCol = np.max(imgData, axis = 0)
        lastRow = index_of_last_nonzero(iRow)
        lastCol = index_of_last_nonzero(iCol)
        firstRow = np.nonzero(iRow)[0][0]
        firstCol = np.nonzero(iCol)[0][0]
        cropped = imgData[firstRow:lastRow,firstCol:lastCol]
        io.imsave(dstDir + "/" + rawFileName, cropped)

        print (f, imgData.shape)



def splitImage(srcDir, dstDir, imageSize) :
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    files = (glob.glob(srcDir + "/*.png"))

    for f in files :
        imgData = io.imread(f)
        rawFileName = ntpath.basename(f)
        newFolderName = dstDir + "/" + rawFileName.split("_")[0]
        if not os.path.exists(newFolderName):
            os.makedirs(newFolderName)

        imgRow = imgData.shape[0]
        imgCol = imgData.shape[1]

        trmRow = imageSize[0]
        trmCol = imageSize[1]

        noOfRow = int ((imgRow + trmRow - 1) / trmRow)
        noOfCol = int  (imgCol  / trmCol)

        startCol = 0
        for c in range(noOfCol) :
            startRow = 0
            r  = 0
            while (startRow + trmRow < imgRow) :
                strOfR = format(r, '02')
                strOfC = format(c, '02')
                newFn = "PAT_" + strOfR + "_" + strOfC + "_" + rawFileName


                cropped = imgData[startRow:startRow + trmRow,startCol : startCol + trmCol]
                iRow = np.min(cropped, axis = 1)
                lastNonZero = index_of_last_zero(iRow)

                if (lastNonZero == -1) :
                    io.imsave(newFolderName + "/" + newFn, cropped)
                    startRow += trmRow
                    r += 1
                else :
                    startRow += lastNonZero + 1

            startCol += trmCol


        print (f, imgData.shape)

def readText():
    dataFolder = "./data"
    trainData = np.loadtxt("./train.txt", dtype="str", delimiter='\t' )
    testData = np.loadtxt("./test.txt", dtype="str",  delimiter='\t' )

    return trainData , testData

def readImage(X_path , y_path):

    for i in range(0,len(X_path)):
        img = cv2.imread (X_path[i])
        X_train.append(cv2.resize(img))
        y_train.append(y_path[i])
    return X_train , y_train



def initEMPercentage(srcDir, trainFile, outputFile) :

    return ;

def tfInit() :
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def filePrefixWith(folder, filePrefix) :
    for filename in os.listdir(folder) :
        if filename.startswith(filePrefix)  :
            return folder + "/" + filename;

    return "";

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

def suffleFile(fn) :
    return ;

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
    return ;

    # Create the model9
def cnn():
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, input_shape=(80, 80, 3), activation= 'relu' , border_mode= 'same' ))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation= 'relu' , W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation= 'relu' , W_constraint=maxnorm(2)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation= 'relu' , W_constraint=maxnorm(1)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation= 'softmax' ))
    model.compile(loss= 'categorical_crossentropy' , optimizer='adam', metrics=[ 'accuracy' ])
    return model;

def main() :

   imagePreprocessing("./data", "./trimDark", "./ImagePatch", "./train.txt", "./ImagePatch/emInit.txt")
   EMLoop("./ImagePatch/emInit.txt", "./ImagePatch", "./ImagePatch/EMFinal.txt", 10, 5);

   tfInit();


    epochs = 200
    estimator = KerasClassifier(build_fn=cnn)


    estimator.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size = 64)


    y_pred = estimator.predict(X_test)

main();
