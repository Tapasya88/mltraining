# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:53:46 2023

@author: Administrator
"""

import pandas as pd
from matplotlib import pyplot as plt
df=pd.read_csv("C:/Users/Administrator/Documents/training/population.csv")

# print(df["ram"])

# plt.hist(df['ram'],bins=25,color='red',edgecolor='blue')
# plt.xlabel('RAM')
# plt.ylabel("# of mobile phones")
# plt.show()


import seaborn as sns
sns.set(style="white", color_codes=True)
sns.barplot(x="Year", y= "Value", data=df.head(n=10))

sns.swarmplot(x="Year", y= "Value", data=df.head(n=10))

sns.stripplot(x="Year", y= "Value", data=df.head(n=10))
