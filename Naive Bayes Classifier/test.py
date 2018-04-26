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

#print(dataset)

# Seperating value into two objects
#X = dataset.iloc[:, [2,3,4]].values
#A = dataset.iloc[:, 17].values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 21].values
#dataset[X[:, 17]] = dataset[X[:, 17]].replace(0, dataset[X[:, 17]].mean())
#dataset[A] = dataset[A].replace(0, dataset[A].mean)

print("This is new age column replace with male and female into 0 and 1.")
#print(A)
print(X)
print(y)

print("This is the new value of gender column with missing values.")
print(X[:, 18])

# Replacing nan value with median in gender column which is 18th column

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'median', axis = 0)
#X[:, [18,19]] = X[:,[18,19]].reshape(1, -1)
imputer = imputer.fit(X[:, [18,19]])
X[:, [18,19]] = imputer.transform(X[:, [18,19]])
print("This is gender column after replacing missing values with median function")
print(X[:, 18])
print("\n")

# Split the data into training and testing part
from sklearn.cross_validation import train_test_split
# Giving 20% of total dataset in testing part and remaining in training part (80%)

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = 0)


# Implementing Naive-Bayes Classifier algorithm for our dataset
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x_train,y_train)

#Predict Output 
predicted= model.predict(x_test)
print("This is the predicted data for testing X which contains 2000 datasets")
print(predicted)
print("\n")

accuracy = model.score(x_test, y_test)
print("This is the accuracy of Naive Bayes Classifier algoritham")
print(accuracy) 
print("\n")

# Calculating confusion matrix for machine learning prediction and 
cm= confusion_matrix(y_test, predicted)
print("The confusion matrix is described below.")
print(cm)



