# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

ds = pd.read_csv("C:/Users/Administrator/Documents/training/churn-bigml-20.csv")
print("Data_frame",ds)
print("Data_frame",ds.shape)
print("Data_frame",ds.columns)
print("Data_frame",ds.index)
print("Total night calls",ds['Total night calls'])
print("Total night calls", ds.head)
print("LOCATION", ds.iloc[3:5,0:2])
print("Count missing data",ds.isnull().sum())