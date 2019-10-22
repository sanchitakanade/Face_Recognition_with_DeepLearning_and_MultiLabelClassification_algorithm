# Name: Sanchita kanade
# Class:CS 596 Machine Learning Fall 2018
# file name :combinePerformance.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy import ndimage
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
import warnings
import loadData as ld

## model no.1 naive_bayes
#loading the data
x_train,y_train,x_validate,y_validate,x_test,y_test = ld.loadData()

# following code creates a classifier and apply it over the validation subset 
clf = MultinomialNB()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_validate)

# following code creates a classifier and apply it over the testing subset 
model = MultinomialNB()
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

#following code claculates accuracy, precision and recall rates.
accuracy_1 = accuracy_score(y_test,y_pred)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_1 = precision_score(y_test,y_pred,average='weighted')
    recall_1 = recall_score(y_test,y_pred,average='weighted')

##################################################################################################################    

# Following code does feature scaling for following models
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)

# model no.2 svm

# following code creates a classifier and apply it over the validation subset 
model = svm.SVC(kernel='linear', C=1.0,gamma =0.01) 
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_validate)

# following code creates a classifier and apply it over the testing subset 
best_kernel = 'linear'
best_c = 1.0
model = svm.SVC(kernel=best_kernel, C=best_c,gamma = 0.01)
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

#following code claculates accuracy, precision and recall rates.
accuracy_2 = accuracy_score(y_test,y_pred)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_2 = precision_score(y_test,y_pred,average='weighted')
    recall_2 = recall_score(y_test,y_pred,average='weighted')

##################################################################################################################
 
# model no.3 logistic regression

# following code creates a classifier and apply it over the validation subset 
clf = LogisticRegression(solver ='lbfgs',max_iter = 1000,multi_class='multinomial')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_validate)

# following code creates a classifier and apply it over the testing subset 
model =LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

#following code claculates accuracy, precision and recall rates.
accuracy_3 = accuracy_score(y_test,y_pred)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_3 = precision_score(y_test,y_pred,average='weighted')
    recall_3 = recall_score(y_test,y_pred,average='weighted')

################################################# FNN ###########

# model no.4 Neural network

# following code creates a classifier and apply it over the validation subset 
clf = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =0.01 ,max_iter = 1500)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_validate)

# following code creates a classifier and apply it over the testing subset 
model = MLPClassifier(hidden_layer_sizes=(80,80),activation = 'logistic',solver ='sgd',learning_rate = 'constant',
                    learning_rate_init =0.01 ,max_iter = 1500)
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

#following code claculates accuracy, precision and recall rates.
accuracy_4 = accuracy_score(y_test,y_pred)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision_4 = precision_score(y_test,y_pred,average='weighted')
    recall_4 = recall_score(y_test,y_pred,average='weighted')

# following code plots a bar graph in order to compare performance of above mentioned algorithms in terms of accuracy. 
objects = ('MultinomialNB','svm','LogisticRegression','MLPClassifier')
y_pos = np.arange(len(objects)) # how many entries on the y axis
performance = [accuracy_1,accuracy_2,accuracy_3,accuracy_4]
 
plt.bar(y_pos, performance, align='center',color='g', alpha = 0.5)
plt.xticks(y_pos, objects)#number of entries on the x axis and their labels
plt.ylabel('Accuracy')
plt.title('Accuracy comparison for all models')
plt.show()

# following code plots a graph in order to compare performance of above mentioned algorithms in terms of precision. 
names = ['MultinomialNB','svm','LogisticRegression','MLPClassifier']
values = [precision_1,precision_2,precision_3, precision_4]

plt.plot(names, values,marker='o')
plt.ylabel('Precision')
plt.title('\nPrecision comparison for all models')
show()

# following code plots a graph in order to compare performance of above mentioned algorithms in terms of recall. 
names = ['MultinomialNB','svm','LogisticRegression','MLPClassifier']
values = [recall_1,recall_2,recall_3, recall_4]

plt.plot(names, values,marker='o')
plt.ylabel('Recall')
plt.title('\nRecall comparison for all models')
show()

