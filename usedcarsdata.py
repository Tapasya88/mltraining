# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

ds = pd.read_csv("C:/Users/Administrator/Documents/training/used_cars_data.csv")
print("Data_frame",ds)
print("Data_frame",ds.shape)
print("Data_frame",ds.columns)
print("Data_frame",ds.index)
print("Total night calls", ds.head)
print("LOCATION", ds.iloc[3:5,0:2])
print("Count missing data",ds.isnull().sum())
print("% Missing data",ds.isnull().sum()/len(ds)*100)

