# Name: Sanchita kanade
# Class:CS 596 Machine Learning Fall 2018
# file name :naive_bayes.py

from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import loadData as ld

# following code loads training, validation and testing data into x_train, y_train,
# x_validate, y_validate, x_test and y_test arrays.
x_train,y_train,x_validate,y_validate,x_test,y_test = ld.loadData()

# creats a classifier  
clf = MultinomialNB()

# training the model
clf.fit(x_train,y_train)

# makes prediction on validation data
y_pred = clf.predict(x_validate)

#creating another classifier for testing the model
model = MultinomialNB()
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

# following code calculates evaluation metrics, including confusion matrix, accuracy, precision and recall rates
print("model:sklearn MultinomialNB")
print("===========================\n")
accuracy_1,precision,recall = ld.func_calConfusionMatrix(y_test,y_pred) 
print("\naverage accuracy:",accuracy_1) 
print("\nper class recall:\n ",recall)
print("\nper class precision:\n",precision)

# following code displays images of wrong predictions
counter = 0
imgArr = []
pred = []
gnd = []
for i in range(len(y_test)):
    if(y_test[i] != y_pred[i] and counter < 10 ):
        imgArr.append(x_test[i]) 
        pred.append(y_pred[i])
        gnd.append(y_test[i])
        counter += 1
w=10                  
h=10
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 1
print("\n\nFailure cases: ")
print("=================")
for i in range(1, columns*rows +1):
    img = imgArr[i]
    img = img.reshape((64,64))
    fig.add_subplot(rows, columns, i)
    rotated_img = ndimage.rotate(img,270)
    plt.imshow(rotated_img,interpolation='nearest')
    plt.title("\nPred: {}  Actual o/p: {}".format(pred[i],gnd[i]),fontsize=11)
    plt.axis('off')
plt.show()

# following code displays images of correct prediction.
counter = 0
imgArr = []
pred = []
gnd = []
for i in range(len(y_test)):
    if(y_test[i] == y_pred[i] and counter < 10 ):
        imgArr.append(x_test[i]) 
        pred.append(y_pred[i])
        gnd.append(y_test[i])
        counter += 1
w=10                  
h=10
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 1
print("\n\nSuccess cases: ")
print("=================")
for i in range(1, columns*rows +1):
    img = imgArr[i]
    img = img.reshape((64,64))
    fig.add_subplot(rows, columns, i)
    rotated_img = ndimage.rotate(img,270)
    plt.imshow(rotated_img,interpolation='nearest')
    plt.title("\nPred: {}  Actual o/p: {}".format(pred[i],gnd[i]),fontsize=11)
    plt.axis('off')
plt.show()
