# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:33:39 2018

@author: patel
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn import preprocessing,cross_validation,svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from sklearn import tree
from matplotlib import style
from matplotlib.colors import ListedColormap
style.use("ggplot")


dataset = pd.read_csv('test.csv')
dataset = dataset.replace(['male','female'],
[0,1])



# Seperating value into two objects
#X = dataset.iloc[:, [2,3,4]].values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 21].values


# Replacing nan value with median in gender column which is 18th column

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'median', axis = 0)
#X[:, [18,19]] = X[:,[18,19]].reshape(1, -1)
imputer = imputer.fit(X[:, [18,19]])
X[:, [18,19]] = imputer.transform(X[:, [18,19]])

print(X[:, 18])



# Normalizing the data from (0 to 1)
X = ((X-X.min())/(X.max()-X.min()))
print(X)

# Split the data into training and testing part
from sklearn.cross_validation import train_test_split
# Giving 20% of total dataset in testing part and remaining in training part (80%)

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = 0)

# Applying K-Neighbour classifier algoritham to our dataset with 5 surrounding neighbors and default matrix 
# and value of p=2 because of this it uses the Euclidean Distance algoritham (2)  
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Giving X_test in predicting the result of y_pred
y_pred = classifier.predict(x_test)
print(y_pred[:])
np.savetxt("y_prediction.csv",y_pred,delimiter=',')

# Calculating the accuracy of the algoritham by comparing the actual target feature and what we received by our machine learning model
accuracy = classifier.score(x_test, y_test)
print("This is the accuracy of our algoritham")
print(accuracy) 

# Calculating confusion matrix for machine learning prediction and 
cm= confusion_matrix(y_test, y_pred)
print("The confusion matrix is described below.")
print(cm)


X_test2 = np.array([0.999999,2.08226e-06,1.93353e-06,	1.93353e-06,	1.93353e-06,	2.13184e-06,	2.231e-06,	4.77028e-05,	3.7679e-06,	3.42086e-06,	9.3206e-06,	9.3206e-06,	1.98311e-06,	0.999032,	0.999037,	1.93353e-06,	1.98311e-06,	1.93353e-06,	1.93353e-06,	2.13184e-06,	0.999538
])
X_test2 = X_test2.reshape(1, -1)
y_pred2 = classifier.predict(X_test2)

print("We inserted query into model and it predicted following decision")
print(y_pred2)



