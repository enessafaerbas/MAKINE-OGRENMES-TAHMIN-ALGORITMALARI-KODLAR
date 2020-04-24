# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:03:19 2020

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm

#2.1 VEri yükleme
veriler = pd.read_csv('maaslar_yeni.csv')
print(veriler)

x=veriler.iloc[:,2:3] # bağımlı değişkenler yani model eğitiminde kullanacağımız değişkenler
y=veriler.iloc[:,-1:] # bağımsız değişkenimiz maaşlardır  
X=x.values             
Y=y.values


print(veriler.corr())

# Yapmış olduğumuz işlemler
# Bağımlı ve Bağımsız değişkenleri  Belirledik hiç bir işlem yapmadan 
# Daha sonra Linear Regresyon ile P value değerlerini hesaplayıp seçtiğimiz bağımlı değişkenlerden eleme yaptık 


# linear Regresyon
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression() 
lin_reg.fit(X,Y)

print("linear Regression tek bağımlı değişken için R-Squared ve p value değerleri")
model1=sm.OLS(lin_reg.predict(X),X)
print(model1.fit().summary())     



# Polynominal Regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2) 
x_poly= poly_reg.fit_transform(X) 
lin_reg2=LinearRegression() 
lin_reg2.fit(x_poly,y)

print("Polynominal Regression tek bağımlı değişken için R-Squared ve p value değerleri")
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())



# öznitelik ölçeklendirme
from sklearn.preprocessing import StandardScaler   

sc1= StandardScaler()
x_olcekli= sc1.fit_transform(X)
sc2= StandardScaler()                          
y_olcekli= sc2.fit_transform(Y)    # DEstek vektörü için gerekli olan aşama Ölçeklendirme



# Destek Vektörü
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')  
svr_reg.fit(x_olcekli,y_olcekli)

print("DEstek Vektörü tek bağımlı değişken için R-Squared ve p value değerleri")
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary()) 



# Karar ağaçları 
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print("Decision Tree tek bağımlı değişken için R-Squared ve p value değerleri")
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary()) 


# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg= RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)

print("Random Forest tek bağımlı değişken için R-Squared ve p value değerleri")
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


# 3 değikenli R - Square değerleri 
 
# linear Regresyon  R-squared:                       0.903
# Polynominal Regression R-squared:                       0.729
# DEstek Vektörü   R-squared:                       0.782
# Decision Tree   R-squared:                       0.679
# Random Forest   R-squared:                       0.713

# TEK değikenli R - Square değerleri

# linear Regresyon   R-squared:                       0.942
# Polynominal Regression  R-squared:                       0.810
# DEstek Vektörü  R-squared:                       0.770
# Decision Tree   R-squared:                       0.751
# Random Forest   R-squared:                       0.719
 












































































