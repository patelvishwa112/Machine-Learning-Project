# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:37:57 2018

@author: patel
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from sklearn import tree
from matplotlib import style
from matplotlib.colors import ListedColormap
style.use("ggplot")

dataset = pd.read_csv('train.csv')
#dataset1 = pd.read_csv('test.csv')
#dataset.iloc[:, 39:42].replace('nan',dataset.iloc[:, 39:42].median(), inplace =True)
X = dataset.iloc[:136, 2:42].values
y = dataset.iloc[:136, 42:].values

#X_test = dataset1.iloc[:36, 3:41].values


#For Multiple Linear Regression
#COnvert categorical data into numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0].astype(str))
X[:, 1] = labelencoder_X.fit_transform(X[:, 1].astype(str))
X[:, 2] = labelencoder_X.fit_transform(X[:, 2].astype(str))

onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()

#Avoid the Dummy Variable Trap
X = X[:, 1:]

#X_test = X_test[:, 1:]

#For Support Vector Machine for Regression

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
#sc_X_test = StandardScaler()
X[:, 38:] = sc_X.fit_transform(X[:, 38:])
y = sc_y.fit_transform(y)
#X_test = sc_X_test.fit_transform(X_test)



# Split the data into training and testing part
from sklearn.cross_validation import train_test_split
# Giving 20% of total dataset in testing part and remaining in training part (80%)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = 0)


# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
#regressor = SVR(kernel = 'rbf')
regressor = SVR(kernel = 'poly')
regressor.fit(X_train, y_train)





y_pred = sc_y.inverse_transform(regressor.predict(X_test))


from sklearn.metrics import r2_score
r2score = r2_score(y_test, y_pred, sample_weight=None, multioutput="uniform_average")

print(r2score)




