# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:25:41 2023

@author: Administrator
"""

import pandas as pd

ds = pd.read_csv("C:/Users/Administrator/Documents/training/Position_Salaries.csv")
df = pd.DataFrame({"Level":ds["Level"], "Salary":ds["Salary"]})
x = df.values
print(df)

import numpy as np
stdev=np.std(df.iloc[:,1])
variance=np.var(df.iloc[:,1])

print("The Salary closer to mean value", stdev)
print("The Salary Variance", variance)

#calculate the percentile
perc =  np.percentile(df.iloc[:,1],25)
print("25th Percentile", perc)