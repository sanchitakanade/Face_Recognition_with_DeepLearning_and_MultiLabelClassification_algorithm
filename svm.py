# Name: Sanchita kanade
# Class:CS 596 Machine Learning Fall 2018
# file name :svm.py

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
import warnings
from sklearn import svm
import loadData as ld
from sklearn.preprocessing import StandardScaler

# following code loads training, validation and testing data into x_train, y_train,
# x_validate, y_validate, x_test and y_test arrays.
x_train,y_train,x_validate,y_validate,x_test,y_test = ld.loadData()

# Following code does feature scaling
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)

print("\n\nmodel: svm")
print("============\n")

# following code plots the validation errors while using different values of C (with other hyperparameters fixed) 
#  keeping kernel = "linear"
c_range = [0.1, 1, 15, 20, 25, 30]
svm_c_error = []
for c_value in c_range:
    model = svm.SVC(kernel='linear', C=c_value,gamma =0.01)
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validate, y_validate)
    svm_c_error.append(error)
plt.plot(c_range, svm_c_error)
plt.title('Linear SVM')
plt.xlabel('c values')
plt.ylabel('error')
plt.show()


# following code plots the validation errors while using linear, RBF kernel, 
# or Polynomial kernel ( with other hyperparameters fixed) 
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
for kernel_value in kernel_types:
    model_2 = svm.SVC(kernel=kernel_value, C=9, gamma ='auto')
    model_2.fit(X=x_train, y=y_train)
    error = 1. - model_2.score(x_validate, y_validate)
    svm_kernel_error.append(error)
    
plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()

#Selecting the best model and apply it over the testing subset 
best_kernel = 'linear'
best_c = 1.0
model = svm.SVC(kernel=best_kernel, C=best_c,gamma = 0.01)
model.fit(X=x_train, y=y_train) #training the model
y_pred = model.predict(x_test)

# following code calculates evaluation metrics, including confusion matrix, accuracy, precision and recall rates

accuracy_2,precision,recall = ld.func_calConfusionMatrix(y_test,y_pred) 
print("\naverage accuracy:",accuracy_2) 
print("\nper class recall:\n ",recall)
print("\n\nper class precision:\n",precision,"\n")

# following code creates and evaluates different models, with different values for hyperparameter gamma.

model_1 = svm.SVC(kernel='linear', C=1.0,gamma = 100000)
model_1.fit(X=x_train, y=y_train)
y_pred_1 = model_1.predict(x_test)

accuracy_1 = accuracy_score(y_test,y_pred_1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_1 = precision_score(y_test,y_pred_1,average ='weighted')
    recall_1 = recall_score(y_test,y_pred_1,average ='weighted')
    
############################################################################

model_2 = svm.SVC(kernel='linear', C=1.0,gamma = 10000)
model_2.fit(X=x_train, y=y_train)
y_pred_2 = model_2.predict(x_test)

accuracy_2 = accuracy_score(y_test,y_pred_2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_2 = precision_score(y_test,y_pred_2,average ='weighted')
    recall_2 = recall_score(y_test,y_pred_2,average ='weighted')

#########################################################################

model_3 = svm.SVC(kernel='linear', C=1.0,gamma = 1000)
model_3.fit(X=x_train, y=y_train)
y_pred_3 = model_3.predict(x_test)

accuracy_3 = accuracy_score(y_test,y_pred_3)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_3 = precision_score(y_test,y_pred_3,average ='weighted')
    recall_3 = recall_score(y_test,y_pred_3,average ='weighted')
    
######################################################################

model_4 = svm.SVC(kernel='linear', C=1.0,gamma = 100)
model_4.fit(X=x_train, y=y_train)
y_pred_4 = model_4.predict(x_test)

accuracy_4 = accuracy_score(y_test,y_pred_4)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_4 = precision_score(y_test,y_pred_4,average ='weighted')
    recall_4 = recall_score(y_test,y_pred_4,average ='weighted')
    
############################################################################
accuracy_5 = accuracy_score(y_test,y_pred)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_5 = precision_score(y_test,y_pred,average ='weighted')
    recall_5 = recall_score(y_test,y_pred,average ='weighted')
#################################################################################

#following code plots a bar graph for various models with different values for hyperparameter gamma.
n_groups = 3

data_1 = (accuracy_1, precision_1, recall_1)
data_2 = (accuracy_2, precision_2, recall_2)
data_3 = (accuracy_3, precision_3, recall_3)
data_4 = (accuracy_4, precision_4, recall_4) 
data_5 = (accuracy_5, precision_5, recall_5)

#following code creates a plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.5

rects1 = plt.bar(index, data_1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='kernel=linear, C=1.0, gamma = 100000')
 
rects2 = plt.bar(index + bar_width, data_2, bar_width,
                 alpha=opacity,
                 color='g',
                 label='kernel=linear, C=1.0, gamma = 10000')

rects3 = plt.bar(index + 2*bar_width, data_3, bar_width,
                 alpha=opacity,
                 color='m',
                 label='kernel=linear, C=1.0, gamma = 1000')

rects4 = plt.bar(index + 3*bar_width, data_4, bar_width,
                 alpha=opacity,
                 color='c',
                 label='kernel=linear, C=1.0, gamma = 100')

rects5 = plt.bar(index + 4*bar_width, data_5, bar_width,
                 alpha=opacity,
                 color='r',
                 label='kernel=linear, C=1.0, gamma = 0.01')
 

plt.ylabel('Performance')
plt.title('\nchange in performance with respect to change gamma values')
plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall'))
 
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(0.01, -0.30),
          fancybox=True, shadow=True)

plt.show()

# following code creates and evaluates different models, with different values for hyperparameter kernel.
model_5 = svm.SVC(kernel='poly', C=1.0,gamma = 0.01)
model_5.fit(X=x_train, y=y_train)
y_pred_5 = model_5.predict(x_test)
accuracy_6 = accuracy_score(y_test,y_pred_5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_6 = precision_score(y_test,y_pred_5,average ='weighted')
    recall_6 = recall_score(y_test,y_pred_5,average ='weighted')

###############################################################################

model_6 = svm.SVC(kernel='rbf', C=1.0,gamma = 0.01)
model_6.fit(X=x_train, y=y_train)
y_pred_6 = model_6.predict(x_test)
accuracy_7 = accuracy_score(y_test,y_pred_6)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_7 = precision_score(y_test,y_pred_6,average ='weighted')
    recall_7 = recall_score(y_test,y_pred_6,average ='weighted')

################################################################################

model_7 = svm.SVC(kernel='sigmoid', C=1.0,gamma = 0.01)
model_7.fit(X=x_train, y=y_train)
y_pred_7 = model_7.predict(x_test)
accuracy_8 = accuracy_score(y_test,y_pred_7)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_8 = precision_score(y_test,y_pred_7,average ='weighted')
    recall_8 = recall_score(y_test,y_pred_7,average ='weighted')
    
#################################################################################   

#following code plots a bar graph for various models with different values for hyperparameter kernel.
n_groups = 3

data_6 = (accuracy_6, precision_6, recall_6)
data_7 = (accuracy_7, precision_7, recall_7)
data_8 = (accuracy_8, precision_8, recall_8)
data_9 = (accuracy_5, precision_5, recall_5) 

# this code creates a graph
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1

rects1 = plt.bar(index, data_6, bar_width,
                 alpha=0.9,
                 color='m',
                 label='kernel=poly, C=1.0, gamma = 0.01')
 
rects2 = plt.bar(index + bar_width, data_7, bar_width,
                 alpha=0.7,
                 color='m',
                 label='kernel=rbf, C=1.0, gamma = 0.01')

rects3 = plt.bar(index + 2*bar_width, data_8, bar_width,
                 alpha=0.40,
                 color='m',
                 label='kernel=sigmoid, C=1.0, gamma = 0.01')

rects4 = plt.bar(index + 3*bar_width, data_9, bar_width,
                 alpha=0.20,
                 color='m',
                 label='kernel=linear, C=1.0, gamma = 0.01')

plt.ylabel('Performance')
plt.title('\nchange in performance with respect to change in kernel values')
plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall'))
 
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(0.01, -0.30),
          fancybox=True, shadow=True)

plt.show()

# following code creates and evaluates different models, with different values for hyperparameter C.
model_8 = svm.SVC(kernel='linear', C=0.01,gamma = 0.01)
model_8.fit(X=x_train, y=y_train)
y_pred_8 = model_8.predict(x_test)
accuracy_9 = accuracy_score(y_test,y_pred_8)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_9 = precision_score(y_test,y_pred_8,average ='weighted')
    recall_9 = recall_score(y_test,y_pred_8,average ='weighted')

#########################################################################
#before this model put your best model  c = 1
model_9 = svm.SVC(kernel='linear', C=100.0,gamma = 0.01)
model_9.fit(X=x_train, y=y_train)
y_pred_9 = model_9.predict(x_test)
accuracy_10 = accuracy_score(y_test,y_pred_9)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_10 = precision_score(y_test,y_pred_9,average ='weighted')
    recall_10 = recall_score(y_test,y_pred_9,average ='weighted')
######################################################################
model_10 = svm.SVC(kernel='linear', C=1000.0,gamma = 0.01)
model_10.fit(X=x_train, y=y_train)
y_pred_10 = model_10.predict(x_test)
accuracy_11 = accuracy_score(y_test,y_pred_10)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_11 = precision_score(y_test,y_pred_10,average ='weighted')
    recall_11 = recall_score(y_test,y_pred_10,average ='weighted')
######################################################################
    
#following code plots a bar graph for various models with different values for hyperparameter C.   
n_groups = 3

data_10 = (accuracy_9, precision_9, recall_9)
data_11 = (accuracy_5, precision_5, recall_5)
data_12 = (accuracy_10, precision_10, recall_10)
data_13 = (accuracy_11, precision_11, recall_11) 

#create a plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1

rects1 = plt.bar(index, data_10, bar_width,
                 alpha=0.9,
                 color='g',
                 label='kernel=linear, C=0.01, gamma = 0.01')
 
rects2 = plt.bar(index + bar_width, data_11, bar_width,
                 alpha=0.7,
                 color='g',
                 label='kernel=linear, C=1.0, gamma = 0.01')

rects3 = plt.bar(index + 2*bar_width, data_12, bar_width,
                 alpha=0.40,
                 color='g',
                 label='kernel=linear, C=100.0, gamma = 0.01')

rects4 = plt.bar(index + 3*bar_width, data_13, bar_width,
                 alpha=0.20,
                 color='g',
                 label='kernel=linear, C=1000.0, gamma = 0.01')

plt.ylabel('Performance')
plt.title('\nchange in performance with respect to change in C values')
plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall'))
 
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(0.01, -0.30),
          fancybox=True, shadow=True)
plt.show()


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
