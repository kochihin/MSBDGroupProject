
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

def fileProcessing():
    dataFolder = "./data"
    trainData = np.loadtxt("./train.txt", dtype="str", delimiter='\t' )
    testData = np.loadtxt("./test.txt", dtype="str",  delimiter='\t' )
    return trainData , testData
	
def filePrefixWith(folder, filePrefix) :
    for filename in os.listdir(folder) : 
        if filename.startswith(filePrefix)  :
            return folder + "/" + filename;

    return "";
