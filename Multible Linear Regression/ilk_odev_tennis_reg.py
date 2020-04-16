# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:49:28 2020

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#tenis oynamak için gerekli bilgilerin olduğu veri kümemizi yükledik. Ve bu kümeden nem oranını tahmin etmemize 
#yardımcı olacak modeli eğiticez çoklu doğrusal regresyon ile 
tenis_veri=pd.read_csv("odev_tenis.csv")
print(tenis_veri)

# öncelikle veri ön işleme aşamasını gerçekleştiricez. kategorik verileri sayısal verilere dönüştürücez.
#veri ön işleme: kategorik verilerin ayrımını yapıcaz

# encoder: kategorik --> Numeric
#play ve windy sütünlarını labelencoder uygulayacağız kukla değişkenden kurtulmak için. outlook a ise ohe uygulayacağız
#bu yöntem uzun bir yöntem daha kısası aşağıda
veriler2=tenis_veri.apply(LabelEncoder().fit_transform)

outlook=veriler2.iloc[:,:1].values
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

#veri ön işleme işlemi bitti şuana kadar apply metodu ile bütün sütünlara laberlencoder uyguladık sonrasında 
#sonrasında outlook sütununa ise onehotencoder uyguladık bundan sonraki işlem verileri bir araya getirmek
#data Frame oluşturmak.

havadurumu= pd.DataFrame(data=outlook, index=range(14), columns=['overcast','rainy','sunny'])
# şimdi ise verileri birleştiricez
sonveriler=pd.concat([havadurumu,tenis_veri.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)
# eğitim verileri ayırma
from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0 )
# nem tahmini için eğitilen model 
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)

# backward elimination 
import statsmodels.formula.api as sm
X= np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=sonveriler.iloc[:,-1:], exog=X_l)
r=r_ols.fit()
print(r.summary())

# enyüksek p değerine sahip olan değeri atıcaz 

sonveriler= sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm
X= np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_l=sonveriler.iloc[:,[0,1,2,3,4,]].values
r_ols=sm.OLS(endog=sonveriler.iloc[:,-1:], exog=X_l)
r=r_ols.fit()
print(r.summary())

# eğitimi ilk sütünü attıktan sonra tekrar yapıcaz X train ve X test de çıkarmamız gerek

x_train= x_train.iloc[:,1:]
x_test= x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)







































