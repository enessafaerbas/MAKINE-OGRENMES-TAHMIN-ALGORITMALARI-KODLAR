# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:19:05 2020

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



# öznitelik ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc1= StandardScaler()
x_olcekli= sc1.fit_transform(X)
sc2= StandardScaler()
y_olcekli= sc2.fit_transform(Y)


# Destek Vektör ile tahmin
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')  # burada kernel yönetemine bağlı olarak RBF fonksiyonunu kullandık. Verilere daha yakın tahminler elde etmek için
svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue') # mavi renkli olan rbf fonkisyon çıktısı
plt.show()

svr_reg1 = SVR(kernel='poly',C=100,gamma='auto', degree=2, epsilon=.1, coef0=1) # burada ise svm ile polinomal yöntemi kullanarak tahmnide bulunduk
svr_reg1.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg1.predict(x_olcekli),color='black') # siyah rekli olan polinomal tahmin çıktısıdır.
























