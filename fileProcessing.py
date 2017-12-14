
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
