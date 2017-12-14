
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
