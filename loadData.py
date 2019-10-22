# Name: Sanchita kanade
# Class:CS 596 Machine Learning Fall 2018
# file name :loadData.py

from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
import warnings

# This function loads the data from two csv files features.csv and labels.csv and
# return arrays for training data, validation data and testing data
def loadData():
    
    #following code loads the data from two csv files and store it into arrays data and labels.
	data = genfromtxt('features.csv', delimiter=',')
	labels = genfromtxt('labels.csv', delimiter=',')

    #following code stores first 165 images from an array data to an array named features.
	features = data[:165,:] 
    
    #function train_test_split the data in the features array into training and testing datasets
	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.30, random_state=None)

	#number of x_train samples = 115
	#number of y_train samples = 115

	#number of x_test samples = 50
	#number of y_test samples = 50

	# transform each image from 64 by64 to a 4096 pixel vector
	pixel_count = 64 * 64
	x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

	nTrain = 80
	
	#following code splits training dataset into training and validation data as follow
    #number of x_train samples = 80
	#number of y_train samples = 80
	#number of x_validation samples = 35
	#number of y_validation samples = 35
	x_train,x_validate = x_train[:nTrain,:], x_train[nTrain:,:]
	y_train,y_validate = y_train[:nTrain],y_train[nTrain:]
	
	return x_train, y_train, x_validate, y_validate, x_test, y_test

# This function calculates confusion matrix, accuracy, per class precision and per class recall 
# and returns accuracy, precision and recall metrics
def func_calConfusionMatrix(trueY,predY):
    cf_matrix = confusion_matrix(trueY, predY)
    print("confusion matrix: \n",cf_matrix)
    accuracy = accuracy_score(trueY,predY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(trueY,predY,average =None)
        recall = recall_score(trueY,predY,average =None)
    return accuracy, precision, recall;

