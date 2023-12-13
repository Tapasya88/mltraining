# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:17:12 2023

@author: Administrator
"""

import pandas as pd
df = pd.read_csv("C:/Users/Administrator/Documents/training/Data.csv")
X= df.iloc[:,:-1].values

print(X)

Y=df.iloc[:,-1].values
print(Y)

#split data for training and test
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.3,random_state=0)
print("** X Training **")
print(X_Train)
print("** X Test **")
print(X_Test)
print("** Y Training **")
print(Y_Train)
print("** Y Test **")
print(Y_Test)

#conversion
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
print(X)

Y=labelencoder.fit_transform(Y)
print(Y)

#onehot indicator
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(X).toarray()
print(x[:,0:4])



