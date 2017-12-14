
import os;

# 3. Import libraries and modules
import numpy as np
#np.random.seed(123)  # for reproducibility
import json

import glob
from sklearn.metrics import classification_report
from skimage import io
from os import listdir
from os.path import isfile, join
import glob
import ntpath


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

def initEMPercentage(srcDir, trainFile, outputFile) :

    trainData = np.loadtxt(trainFile, dtype="str", delimiter='\t' );

    f = open( outputFile, 'w')
    for k in range(len(trainData)) :

        aLine = trainData[k];
        targetDir = srcDir + "/"  + aLine[0]
        label = aLine[1]

        files = (glob.glob(targetDir + "/*.png"))

        totalFile = 0
        fileLen= len(files)
        for file in files :
            f.write(file + "\t" + aLine[1] + "\t" + str( float(aLine[1]) / float(fileLen)) + "\n" )  # python will convert \n to os.linesep

    f.close()  # you can omit in most cases as the destructor will call it

def imagePreprocessing(dataFolder, trimDarkFolder, imagePatchFolder, trainFN, packFN):
    image_size = (224,224)

    print ("Cut the dark border for each image")
    cutDark(dataFolder,trimDarkFolder);

    print ("Split the image into small image with size ", image_size)
    splitImage(trimDarkFolder, imagePatchFolder, image_size)


    print ("Assign label to small image ", image_size)
    initEMPercentage(imagePatchFolder, trainFN, packFN)
