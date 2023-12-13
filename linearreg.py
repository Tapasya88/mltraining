# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:31:33 2018

@author: Balasubramaniam
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
# Importing the dataset
dataset = pd.read_csv('MSFT.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 3].values
print(X,y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print("X Training data")
print(X_train[0:5])

print("X Test data")
print(X_test[0:5])

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#pdb.set_trace()
# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Mean Sqaured Error
MSE=np.mean((y_test-y_pred)**2)
print("Mean Sqaured Error %r" %(MSE))
#Sum of Sqaured Error
SSE=np.sum((y_test-y_pred)**2)
print("SUM of Sqaured Error %r" %(SSE))
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Open vs Close (Training set)')
plt.xlabel('Opening Stock')
plt.ylabel('Closing Stock')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Open vs Close(Test set)')
plt.xlabel('Opening Stock')
plt.ylabel('Closing Stock')
plt.show()