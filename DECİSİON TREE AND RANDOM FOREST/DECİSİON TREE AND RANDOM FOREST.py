# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:56:27 2020

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.1 VEri yükleme
veriler = pd.read_csv('maaslar.csv')
print(veriler)

# data frame dilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,-1:]

# numpy dizi(array) dönüşümü
X=x.values
Y=y.values


# Karar ağaçları 
from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X),color='blue')
plt.show()

# Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf_reg= RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,rf_reg.predict(X),color='blue')












































