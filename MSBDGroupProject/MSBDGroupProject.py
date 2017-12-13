
import numpy as np



dataFolder = "./data"
trainData = np.loadtxt("./train.txt", dtype="str", delimiter='\t' )
testData = np.loadtxt("./test.txt", dtype="str",  delimiter='\t' )
