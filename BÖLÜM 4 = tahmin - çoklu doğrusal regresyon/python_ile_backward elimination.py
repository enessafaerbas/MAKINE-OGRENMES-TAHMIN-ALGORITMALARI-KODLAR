# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:04:12 2020

@author: Enes Safa
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')
print(veriler)

# verileri kategorik ve sayısal olmak üzere ayırmak için yapılan işlem
ulke= veriler.iloc[:,0:1].values
print(ulke)

Yas= veriler.iloc[:,1:4].values
print(Yas)


# encoder: kategorik --> Numeric
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
le=LabelEncoder()
ulke[:,0]= le.fit_transform(ulke[:,0])
print(ulke)

ohe= OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#cinsiyet kategorik bir veri olduğu için ayırıp sayısal değerlere dönüştürdük
c= veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
le=LabelEncoder()
c[:,0]= le.fit_transform(c[:,0])
print(c)

ohe= OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

# data frame oluşturma 
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index= range(22), columns=['boy','kil0','yas'])
print(sonuc2)

cinsiyet= veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3= pd.DataFrame(data = c[:,:1], index =range(22), columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)


# eğitim verileri ayırma
from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0 )

# cinsiyet tahmini için eğitilen model 
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

# boy tahmini için s2 data frame inden boy bilgilerini çekmemiz gerekli.
boy= s2.iloc[:,3:4].values
print(boy)

# boy haric diğer verileri boy kolonunun sağı ve solu olmak üzere bölüdük ve birleştirdik
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]
veri=pd.concat([sol,sag],axis=1)

# eğitim ve test verilerini ayırıp eğitimi gerçekleştirdik
x_train, x_test, y_train , y_test = train_test_split(veri,boy,test_size=0.33, random_state=0 )
r2 =LinearRegression()
r2.fit(x_train,y_train)
y_pred2 = r2.predict(x_test)

# geri eleme yönteminin başlangıcı
# formül deki sabit değeri eklemek için yapılan işlem 22 tane 1 den oluşan dizi oluşturmak formüldeki beta 0 değeri

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)



































