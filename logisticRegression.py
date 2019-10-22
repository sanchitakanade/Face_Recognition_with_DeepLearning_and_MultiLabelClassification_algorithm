# Name: Sanchita kanade
# Class:CS 596 Machine Learning Fall 2018
# file name :logisticRegression.py

import loadData as ld
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
clf = LogisticRegression(solver ='lbfgs',max_iter = 1000,multi_class='multinomial')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_validate)

# following code creates a classifier and apply it over the testing subset 
model =LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000)
model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

# following code calculates evaluation metrics, including confusion matrix, accuracy, precision and recall rates
print("\n\nmodel: LogisticRegression")
print("===========================\n")

accuracy_3,precision,recall = ld.func_calConfusionMatrix(y_test,y_pred) 
print("\naverage accuracy:",accuracy_3) 
print("\nper class recall:\n ",recall)
print("\nper class precision:\n",precision)

##following code creates different models in order to evaluate the performance in
# terms of accuracy, precision and recall by changing different hyperparameters
accuracyGraphEntry_5 = accuracy_3
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precisionGraphEntry_5 = precision_score(y_test,y_pred,average = 'weighted')
    recallGraphEntry_5 = recall_score(y_test,y_pred,average = 'weighted')

model_2 =LogisticRegression(solver = 'saga', multi_class = 'multinomial', max_iter = 4000)
model_2.fit(X=x_train, y=y_train)
y_pred2 = model_2.predict(x_test)
accuracyGraphEntry_1 = accuracy_score(y_test, y_pred2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precisionGraphEntry_1 = precision_score(y_test,y_pred2,average = 'weighted')
    recallGraphEntry_1 = recall_score(y_test,y_pred2,average = 'weighted')

model_3 =LogisticRegression(solver = 'sag', multi_class = 'multinomial', max_iter = 3000)
model_3.fit(X=x_train, y=y_train)
y_pred3 = model_3.predict(x_test)
accuracyGraphEntry_2 = accuracy_score(y_test, y_pred3)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precisionGraphEntry_2 = precision_score(y_test,y_pred3,average = 'weighted')
    recallGraphEntry_2 = recall_score(y_test,y_pred3,average = 'weighted')

model_4 =LogisticRegression(solver = 'sag', multi_class = 'multinomial', max_iter = 2500)
model_4.fit(X=x_train, y=y_train)
y_pred4 = model_4.predict(x_test)
accuracyGraphEntry_3 = accuracy_score(y_test, y_pred4)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precisionGraphEntry_3 = precision_score(y_test,y_pred4,average = 'weighted')
    recallGraphEntry_3 = recall_score(y_test,y_pred4,average = 'weighted')

model_5 =LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 2000)
model_5.fit(X=x_train, y=y_train)
y_pred5 = model_5.predict(x_test)
accuracyGraphEntry_4 = accuracy_score(y_test, y_pred5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precisionGraphEntry_4 = precision_score(y_test,y_pred5,average = 'weighted')
    recallGraphEntry_4 = recall_score(y_test,y_pred5,average = 'weighted')

# following code plots a bar graph for above models
entry1 = []
entry2 = []
entry3 = []
entry4 = []
entry5 = []
y_axisLabels = ['Accuracy', 'Precision', 'Recall']
entry1.extend([accuracyGraphEntry_1,precisionGraphEntry_1,recallGraphEntry_1])  ## add list of elems at end
entry2.extend([accuracyGraphEntry_2,precisionGraphEntry_2,recallGraphEntry_2])  ## add list of elems at end
entry3.extend([accuracyGraphEntry_3,precisionGraphEntry_3,recallGraphEntry_3])  ## add list of elems at end
entry4.extend([accuracyGraphEntry_4,precisionGraphEntry_4,recallGraphEntry_4])  ## add list of elems at end
entry5.extend([accuracyGraphEntry_5,precisionGraphEntry_5,recallGraphEntry_5])  ## add list of elems at end

ind = [x for x, _ in enumerate(y_axisLabels)]

entry1 = np.array(entry1)
entry2 = np.array(entry2)
entry3 = np.array(entry3)
entry4 = np.array(entry4)
entry5 = np.array(entry5)

plt.bar(ind, entry1, width=0.8, label='solver = saga, multi_class = multinomial, max_iter = 4000', color='c', bottom=entry2+entry3+entry4+entry5, alpha= 1.0,edgecolor='black', linewidth=1)
plt.bar(ind, entry2, width=0.8, label='solver = sag, multi_class = multinomial, max_iter = 3000', color='c', bottom=entry3+entry4+entry5, alpha = 0.8,edgecolor='black', linewidth=1)
plt.bar(ind, entry3, width=0.8, label='solver = sag, multi_class = multinomial, max_iter = 2500', color='c', bottom=entry4+entry5, alpha = 0.60,edgecolor='black', linewidth=1)
plt.bar(ind, entry4, width=0.8, label='solver = lbfgs, multi_class = multinomial, max_iter = 2000', color='c',bottom=entry5, alpha = 0.40,edgecolor='black', linewidth=1)
plt.bar(ind, entry5, width=0.8, label='solver = lbfgs, multi_class = multinomial, max_iter = 1000', color='c',alpha = 0.2,edgecolor='black', linewidth=1)

plt.xticks(ind, y_axisLabels)
plt.ylabel("Performance")
plt.legend(loc='center left', bbox_to_anchor=(0.01, -0.30),
          fancybox=True, shadow=True)
plt.title("Accuracy, precision, and Recall with different parameters")

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

