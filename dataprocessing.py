# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:07:22 2020

@author: pmo7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('/home/pmo7/Machine Learning /ml program/Data.csv')
X = dataset.iloc[:, :-1].values  #independent Variables
y = dataset.iloc[:, 3].values  #dependend variables

#taking care of missing data
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])

#Encoding Ctagorical Data
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


#splitting dataset into the Training set and test set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)    
