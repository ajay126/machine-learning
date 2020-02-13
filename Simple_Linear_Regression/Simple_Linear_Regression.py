#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values #independent Variabel
y = dataset.iloc[:,1].values

#spliting dataset into traning and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)


#feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_predict = regressor.predict(X_test)    

# Visualising the training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Expreince VS Salary (Traning Set)")
plt.xlabel("Year Of Expreince") 
plt.ylabel("Salary")

plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Expreince VS Salary (Test Set)")
plt.xlabel("Year Of Expreince") 
plt.ylabel("Salary")

plt.show()