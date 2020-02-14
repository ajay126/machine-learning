# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:31:21 2020

@author: DELL
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values #independent variables
y = dataset.iloc[:,4].values #dependend varivables

#categorical data
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])])
X = ct.fit_transform(X)

#spliting dataset into traning and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,xx,color='blue')
plt.title("Profite")
plt.xlabel("Profite")
plt.ylabel("Marketing Spend")
plt.show()

#Checking the score  
print('Train Score: ', regressor.score(X_train, y_train))  
print('Test Score: ', regressor.score(X_test, y_test))  
