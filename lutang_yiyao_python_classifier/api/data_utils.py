from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

class data_utils_method():

    def __init__(self):
        self.Name = "特征工程"
        pass

    def datachange(self,data_load):

        #如果包含Survived则输入的是训练集，取y切掉标签；不包含，则输入的是测试集，不用管y,直接特征工程
        if "target" in data_load.columns:
            ydata = data_load["target"]
            del data_load["target"]
        else:
            ydata = []
 
        #LabelEncoder
        le = LabelEncoder() 
        #data_load.loc[:,'Sex'] = le.fit_transform(data_load['Sex']) #姓名标签
        #data_load.loc[:,'Embarked'] = data_load.loc[:,'Embarked'].apply(lambda x: str(x))#全转成字符串，不然报错
        #data_load.loc[:,'Embarked'] = le.fit_transform(data_load['Embarked'])#Embarked标签
       

        #删掉不要的
        #del data_load["PassengerId"]
        #del data_load["Name"]
        #del data_load["Ticket"]
        #del data_load["Cabin"]
        
        #均值填充
        si = SimpleImputer(missing_values=np.nan,strategy="mean")
        xdata = si.fit_transform(data_load)
        #标准归一化
        sc = StandardScaler()
        xdata = sc.fit_transform(xdata)
        
        
        return xdata,ydata

