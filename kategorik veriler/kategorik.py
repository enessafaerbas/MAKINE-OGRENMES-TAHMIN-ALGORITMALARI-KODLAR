# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:20:33 2019

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler= pd.read_csv("eksikveriler.csv")

ulke= veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

ulke[:,0]= le.fit_transform(ulke[:,0])
print(ulke)
'''
from sklearn.preprocessing import OneHotEncoder

ohe= OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
'''