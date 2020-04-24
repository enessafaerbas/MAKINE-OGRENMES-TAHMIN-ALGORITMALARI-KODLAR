# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:52:51 2020

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


#2.1 VEri yükleme
veriler = pd.read_csv('maaslar.csv')
print(veriler)

# data frame dilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,-1:]

# numpy dizi(array) dönüşümü
X=x.values
Y=y.values

#linear regression doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,Y)

                      ######################## R2 değerinin hesaplanması ######################
print("Linear Regresyon R2 degeri:")
print(r2_score(Y,lin_reg.predict(X)))

#polynominal regression 2. dereceden
# doğrusal olmayan model oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2) #  formülde x değerlerinin 2. derereceden değerlerinin alınması için dereceyi 2 yaptık
x_poly= poly_reg.fit_transform(X) # linear olan değerleri polinom karşılıklarını elde etek için yapılan işlem
lin_reg2=LinearRegression() # tekrar linear regression objesi oluşturduk
lin_reg2.fit(x_poly,y) # polinomal değerleri kullanrak linear regresyon uygulayacağız (x_poly değerlerini kullanarak y değerlerinin öğren) 

                     ######################## R2 değerinin hesaplanması ######################
print("Polynominal R2 degeri:")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

# 4. dereceden polinom
poly_reg3=PolynomialFeatures(degree=4) #  formülde x değerlerinin 4. derereceden alara tahminleri daha başarılı kıldık
x_poly3= poly_reg3.fit_transform(X) 
lin_reg3=LinearRegression() 
lin_reg3.fit(x_poly3,y)  

# görselleştirme

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X))
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()

plt.scatter(X,Y,color='red')  # nokta nokta olarak   x y değerlerin çizdirdik
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color='blue') # burda x in polinomal değerlerini vererek y değerlerini tahmin ettirdik
plt.show()


# tahminler

'''
print(lin_reg.predict(11)) # bu kodlar linear olarak  eğitim seviyesini 11 alarak maaşını tahmin edecek 
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11))) # burada ise 11 değerinin polinomal karşılığını elde edip tahmini ona göre yapacak.
print(lin_reg2.predict(poly_reg.fit.transform(6.6)))

'''


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
'''
svr_reg1 = SVR(kernel='poly',C=100,gamma='auto', degree=2, epsilon=.1, coef0=1) # burada ise svm ile polinomal yöntemi kullanarak tahmnide bulunduk
svr_reg1.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg1.predict(x_olcekli),color='black') # siyah rekli olan polinomal tahmin çıktısıdır.
plt.show()
'''
                          ######################## R2 değerinin hesaplanması ######################
print("SVR (Destek Vektörü) R2 degeri:")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))



# Karar ağaçları 
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X),color='blue')
plt.show()
                         ######################## R2 değerinin hesaplanması ######################
print("Decision Tree R2 degeri:")
print(r2_score(Y,r_dt.predict(X)))



# Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf_reg= RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(x,rf_reg.predict(X),color='blue')
plt.show()

                         ######################### R2 değerinin hesaplanması ###################### import kısmı yukarıda
print("Random Forest R2 degeri:")
print(r2_score(Y,rf_reg.predict(X) ))

# Özet R2 Degerleri
print("-------------------------------------------")

print("Linear Regresyon R2 degeri:")
print(r2_score(Y,lin_reg.predict(X)))

print("Polynominal R2 degeri:")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

print("SVR (Destek Vektörü) R2 degeri:")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("Decision Tree R2 degeri:")
print(r2_score(Y,r_dt.predict(X)))

print("Random Forest R2 degeri:")
print(r2_score(Y,rf_reg.predict(X) ))






















