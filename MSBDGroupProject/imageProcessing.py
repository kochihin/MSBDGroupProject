
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
