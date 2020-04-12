# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:01:42 2019

@author: Enes Safa
"""
#kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yükleme
veriler= pd.read_csv("veriler.csv")
print(veriler)

# veri ön işleme
boy= veriler[['boy']]
print(boy)
boykilo= veriler[['boy','kilo']]
print(boykilo)

class insan:
    boy=180
    def  kosmak(self,b):
        return b+10
    
ali= insan()

print(ali.boy)
print(ali.kosmak(10))