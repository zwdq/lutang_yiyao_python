# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:56:37 2020

@author: lu.tang
"""
import pandas as pd
x_train = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data17-19\X.csv",index_col=0)
y_train_label = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data17-19\y.csv",index_col=0)
x_train["target"] = y_train_label