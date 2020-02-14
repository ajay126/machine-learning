#importing librabys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mpl_toolkits

#importing dataset
dataset = pd.read_csv("/home/pmo7/Machine Learning /ml program/Simple_Linear_Regression/SML1/kc_house_data.csv")
X = dataset.iloc[:,5].values
y = dataset.iloc[:,2].values
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

val = dataset[["price","bedrooms","bathrooms","sqft_living","sqft_lot","sqft_above","yr_built","sqft_living15","sqft_lot15"]].describe()
print(val)
#splitting dataset into the Training set and test set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Visualizing the training Test Results 
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train),color="blue")
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results 
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()