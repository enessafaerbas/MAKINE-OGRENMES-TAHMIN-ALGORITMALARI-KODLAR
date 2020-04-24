# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:22:26 2020

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.1 VEri yükleme
veriler = pd.read_csv('maaslar.csv')
print(veriler)

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,-1:]
X=x.values
Y=y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X))
plt.show()

#polynominal regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2) #  formülde x değerlerinin 2. derereceden değerlerinin alınması için dereceyi 2 yaptık
x_poly= poly_reg.fit_transform(X) # linear olan değerleri polinom karşılıklarını elde etek için yapılan işlem
print(x_poly)
lin_reg2=LinearRegression() # tekrar linear regression objesi oluşturduk
lin_reg2.fit(x_poly,y) # polinomal değerleri kullanrak linear regresyon uygulayacağız (x_poly değerlerini kullanarak y değerlerinin öğren) 

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4) #  formülde x değerlerinin 4. derereceden alara tahminleri daha başarılı kıldık
x_poly= poly_reg.fit_transform(X) 
print(x_poly)
lin_reg2=LinearRegression() 
lin_reg2.fit(x_poly,y)  

plt.scatter(X,Y,color='red')  # nokta nokta olarak   x y değerlerin çizdirdik
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue') # burda x in polinomal değerlerini vererek y değerlerini tahmin ettirdik
plt.show()


# tahminler

'''
print(lin_reg.predict(11)) # bu kodlar linear olarak  eğitim seviyesini 11 alarak maaşını tahmin edecek 
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11))) # burada ise 11 değerinin polinomal karşılığını elde edip tahmini ona göre yapacak.
print(lin_reg2.predict(poly_reg.fit.transform(6.6)))

'''
























