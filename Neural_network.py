# Name: Sanchita kanade
# Class:CS 596 Machine Learning Fall 2018
# file name :Neural_network.py

import loadData as ld
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings

#loading the data
x_train,y_train,x_validate,y_validate,x_test,y_test = ld.loadData()

# Following code does feature scaling
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)

# following code creates a classifier and apply it over the validation subset 
clf = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =0.01 ,max_iter = 2000)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_validate)

# following code creates a classifier and apply it over the testing subset 
model = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =0.01 ,max_iter = 2000)
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

# following code calculates evaluation metrics, including confusion matrix, accuracy, precision and recall rates
print("\n\nmodel: feedforward neural network-FNN")
print("=======================================\n")

accuracy_4,precision,recall = ld.func_calConfusionMatrix(y_test,y_pred) 

print("\naverage accuracy:",accuracy_4) 
print("\nper class recall:\n\n ",recall)
print("\nper class precision:\n",precision,"\n\n")


#following code creates different models in order to observe the performance of different models having different 
#values for hyperparameter hidden_layer_sizes
accuracy_1 = accuracy_score(y_test,y_pred)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_1 = precision_score(y_test,y_pred,average ='weighted')
    recall_1 = recall_score(y_test,y_pred,average ='weighted')

#################################################################################################################
model_1 = MLPClassifier(hidden_layer_sizes=(50,50),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                        learning_rate_init =0.01 ,max_iter = 2000)

model_1.fit(X=x_train, y=y_train)
y_pred_1 = model_1.predict(x_test)

accuracy_2 = accuracy_score(y_test,y_pred_1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_2 = precision_score(y_test,y_pred_1,average ='weighted')
    recall_2 = recall_score(y_test,y_pred_1,average ='weighted')

##########################################################################################################
model_2 = MLPClassifier(hidden_layer_sizes=(30,30),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                        learning_rate_init =0.01 ,max_iter = 2000)

model_2.fit(X=x_train, y=y_train)
y_pred_2 = model_2.predict(x_test)

accuracy_3 = accuracy_score(y_test,y_pred_2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_3 = precision_score(y_test,y_pred_2,average ='weighted')
    recall_3 = recall_score(y_test,y_pred_2,average ='weighted')

############################################################################################################
model_3 = MLPClassifier(hidden_layer_sizes=(100),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                        learning_rate_init =0.01 ,max_iter = 2000)

model_3.fit(X=x_train, y=y_train)
y_pred_3 = model_3.predict(x_test)

accuracy_4 = accuracy_score(y_test,y_pred_3)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_4 = precision_score(y_test,y_pred_3,average ='weighted')
    recall_4 = recall_score(y_test,y_pred_3,average ='weighted')
#################################################################################################

model_4 = MLPClassifier(hidden_layer_sizes=(40,20,30),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                        learning_rate_init =0.1 ,max_iter = 2000)

model_4.fit(X=x_train, y=y_train)
y_pred_4 = model_4.predict(x_test)

accuracy_5 = accuracy_score(y_test,y_pred_4)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_5 = precision_score(y_test,y_pred_4,average ='weighted')
    recall_5 = recall_score(y_test,y_pred_4,average ='weighted')
#########################################################################################################

#following code plots a bar graph for various models with different values for hyperparameter hidden_layer_sizes.

n_groups = 3

data_1 = (accuracy_1, precision_1, recall_1)
data_2 = (accuracy_2, precision_2, recall_2)
data_3 = (accuracy_3, precision_3, recall_3)
data_4 = (accuracy_4, precision_4, recall_4)
data_5 = (accuracy_5, precision_5, recall_5)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.7

rects1 = plt.bar(index, data_1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='hidden_layer_sizes=(80,80), learning_rate_init =0.01 ,max_iter = 2000')
 
rects2 = plt.bar(index + bar_width, data_2, bar_width,
                 alpha=opacity,
                 color='g',
                 label='hidden_layer_sizes=(50,50), learning_rate_init =0.01 ,max_iter = 2000')

rects3 = plt.bar(index + 2*bar_width, data_3, bar_width,
                 alpha=opacity,
                 color='m',
                 label='hidden_layer_sizes=(30,30), learning_rate_init =0.01 ,max_iter = 2000')

rects4 = plt.bar(index + 3*bar_width, data_4, bar_width,
                 alpha=opacity,
                 color='c',
                 label='hidden_layer_sizes=(100), learning_rate_init =0.01 ,max_iter = 2000')

rects5 = plt.bar(index + 4*bar_width, data_5, bar_width,
                 alpha=opacity,
                 color='r',
                 label='hidden_layer_sizes=(40,20,30), learning_rate_init =0.1 ,max_iter = 2000')


plt.ylabel('Performance')
plt.title('change in the performance with respect to change in hidden layers')
plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall'))
 
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(0.01, -0.30),
          fancybox=True, shadow=True)

plt.show()

#following code creates different models in order to observe the performance of different models having different 
#values for hyperparameter activation and learning rate.
model_5 = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'tanh',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =0.01 ,max_iter = 2000)

model_5.fit(X=x_train, y=y_train)
y_pred_5 = model_5.predict(x_test)

accuracy_6 = accuracy_score(y_test,y_pred_5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_6 = precision_score(y_test,y_pred_5,average ='weighted')
    recall_6 = recall_score(y_test,y_pred_5,average ='weighted')
    
#########################################################################################

model_6 = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'relu',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =0.01 ,max_iter = 2000)

model_6.fit(X=x_train, y=y_train)
y_pred_6 = model_6.predict(x_test)

accuracy_7 = accuracy_score(y_test,y_pred_6)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_7 = precision_score(y_test,y_pred_6,average ='weighted')
    recall_7 = recall_score(y_test,y_pred_6,average ='weighted')
    
##########################################################################################

model_7 = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =10 ,max_iter = 2000)

model_7.fit(X=x_train, y=y_train)
y_pred_7 = model_7.predict(x_test)

accuracy_8 = accuracy_score(y_test,y_pred_7)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_8 = precision_score(y_test,y_pred_7,average ='weighted')
    recall_8 = recall_score(y_test,y_pred_7,average ='weighted')

##########################################################################################################

model_8 = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =1 ,max_iter = 2000)

model_8.fit(X=x_train, y=y_train)
y_pred_8 = model_8.predict(x_test)

accuracy_9 = accuracy_score(y_test,y_pred_8)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_9 = precision_score(y_test,y_pred_8,average ='weighted')
    recall_9 = recall_score(y_test,y_pred_8,average ='weighted')
    
######################################################################################################

model_9 = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =0.1 ,max_iter = 2000)

model_9.fit(X=x_train, y=y_train)
y_pred_9 = model_9.predict(x_test)

accuracy_10 = accuracy_score(y_test,y_pred_9)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_10= precision_score(y_test,y_pred_9,average ='weighted')
    recall_10 = recall_score(y_test,y_pred_9,average ='weighted')
    
#following code plots a bar graph for above models with different values for hyperparameter activation and 
#learning rate.
n_groups = 3

data_6 = (accuracy_6, precision_6, recall_6)
data_7 = (accuracy_7, precision_7, recall_7)
data_8 = (accuracy_1, precision_1, recall_1)
data_9 = (accuracy_8, precision_8, recall_8) 
data_10 = (accuracy_9, precision_9, recall_9)
data_11 = (accuracy_10, precision_10, recall_10)

#creates a plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.5

rects6 = plt.bar(index, data_6, bar_width,
                 alpha=opacity,
                 color='b',
                 label='hidden_layer_sizes=(80,80),activation = tanh, learning_rate_init =0.01, max_iter = 2000')

rects7 = plt.bar(index + bar_width, data_7, bar_width,
                 alpha=opacity,
                 color='g',
                 label='hidden_layer_sizes=(80,80), activation = relu, learning_rate_init =0.01, max_iter = 2000')

rects8 = plt.bar(index + 2*bar_width, data_8, bar_width,
                 alpha=opacity,
                 color='m',
                 label='hidden_layer_sizes=(80,80), activation = logistic, learning_rate_init =0.01, max_iter = 2000')

rects9 = plt.bar(index + 3*bar_width, data_9, bar_width,
                 alpha=opacity,
                 color='c',
                 label='hidden_layer_sizes=(80,80), activation = logistic, learning_rate_init =10, max_iter = 2000')

rects10 = plt.bar(index + 4*bar_width, data_10, bar_width,
                 alpha=opacity,
                 color='r',
                 label='hidden_layer_sizes=(80,80), activation = logistic, learning_rate_init =1, max_iter = 2000')

rects11 = plt.bar(index + 5*bar_width, data_11, bar_width,
                 alpha=opacity,
                 color='gold',
                 label='hidden_layer_sizes=(80,80), activation = logistic, learning_rate_init =0.1, max_iter = 2000')

plt.ylabel('Performance')
plt.title('\nchange in the performance with respect to\nchange in activation and learning rate')
plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall'))
 

plt.legend(loc='center left', bbox_to_anchor=(0.01, -0.40),
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
