# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:42:25 2023

@author: Administrator
"""

import pandas as pd

ds = pd.read_csv("C:/Users/Administrator/Documents/training/Position_Salaries.csv")
df = pd.DataFrame({"Level":ds["Level"], "Salary":ds["Salary"]})
x = df.values
print(df)

from sklearn.preprocessing import Normalizer
scaler = Normalizer()
sclaed_data = scaler.fit_transform(x)
df =  pd.DataFrame(sclaed_data)
print(df)

from sklearn import preprocessing
min_max_scaler =  preprocessing.MinMaxScaler()
x_scaled= min_max_scaler.fit_transform(x)
df =  pd.DataFrame(x_scaled)
print(df)


from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler()
scaled_data2 = stscaler.fit_transform(x)
df =  pd.DataFrame(scaled_data2)
print(df)
