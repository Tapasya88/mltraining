# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:53:09 2023

@author: Administrator
"""

import pandas as pd
from matplotlib import pyplot as plt

ds = pd.read_csv("C:/Users/Administrator/Documents/training/Position_Salaries.csv")
df = pd.DataFrame(ds)
level = df['Level']
salary = ds['Salary']

fig = plt.figure(figsize=(10,7))
plt.bar(level, salary, color="orange")
plt.title("Level vs Salary")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.grid()
plt.show()
