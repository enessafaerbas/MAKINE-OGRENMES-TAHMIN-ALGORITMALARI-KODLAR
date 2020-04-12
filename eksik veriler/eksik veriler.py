# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:14:34 2019

@author: Enes Safa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# eksik veriler

veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

# sci - kit learn
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

Yas= veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]= imputer.transform(Yas[:,1:4])
print(Yas)
