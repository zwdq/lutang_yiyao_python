# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:56:37 2020

@author: lu.tang
"""
import pandas as pd
x_train = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data\x_train.csv",index_col=0)
x_test = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data\x_test.csv",index_col=0)
y_train = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data\y_train.csv",index_col=0)
y_test = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data\y_test.csv",index_col=0)
y_train_lable = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data\y_train_lable.csv",index_col=0)
y_test_lable = pd.read_csv(r"C:\Users\lu.tang\Desktop\28天预测\data\y_test_lable.csv",index_col=0)